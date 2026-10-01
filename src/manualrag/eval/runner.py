"""Run A–E with fixed splits; no silent substitutes for unavailable models."""
from pathlib import Path
from time import perf_counter
import json
import logging
import re
from manualrag.config import Settings
from manualrag.conversation import load_conversation,save_turn
from manualrag.eval.dataset import load_dataset,Question
from manualrag.eval.judge import judge_answer
from manualrag.eval.metrics import retrieval_metrics,citation_metrics
from manualrag.eval.report import write_report
from manualrag.index.store import Store
from manualrag.ingest.error_codes import CODE_PATTERN
from manualrag.llm.client import AnthropicClient
from manualrag.llm.caching import atomic_text
from manualrag.models import ActiveContext,Answer,GateDecision,Hit,QueryPlan,Step,Turn
from manualrag.retrieval.gate import gate,relevance_score,calibrate_threshold
from manualrag.retrieval.hybrid import Retriever,TEXT_TYPES,same_section
from manualrag.service import ManualAssistant

logger=logging.getLogger(__name__)
ABLATIONS={"A":("bm25",TEXT_TYPES,False),"B":("dense",TEXT_TYPES,False),"C":("hybrid",TEXT_TYPES,False),
           "D":("hybrid",TEXT_TYPES|{"figure","page_card"},False),"E":("hybrid",TEXT_TYPES|{"figure","page_card"},True)}

def retrieval_plan(q: Question,model: str) -> QueryPlan:
    """Use a documented history-concatenation baseline for API-free retrieval evaluation."""
    query=q.question
    if q.history:query=" ".join(h.question+" "+h.answer for h in q.history)+" "+query
    match=CODE_PATTERN.search(q.question)
    error=match.group() if match and not match.group().upper().startswith(model.split("-")[0].upper()) else None
    prefix=re.match(r"[A-Za-z]+",model)
    other=None
    if prefix:
        models=re.findall(r"\b"+re.escape(prefix.group())+r"[- ]?\d+\b",q.question,re.I)
        other=next((m for m in models if m.replace("-","").replace(" ","").casefold()!=model.replace("-","").replace(" ","").casefold()),None)
    return QueryPlan(standalone_query=query,intent="error_code" if error else "spec_lookup",
                     error_code=error,needs_visual=q.needs_visual,other_model_mentioned=other)

def seed_history(store: Store,product_id: str,q: Question,settings: Settings) -> str | None:
    """Seed supplied history with valid manual references; the next turn re-retrieves."""
    if not q.history:return None
    cid,_,_=load_conversation(store,product_id,None,settings.history_turns)
    chunks=store.chunks(product_id)
    for h in q.history:
        matches=[c for c in chunks if c.page_no in h.pages]
        steps=[Step(n=1,text=h.answer,chunk_ids=[matches[0].chunk_id])] if matches else []
        answer=Answer(status="answered" if steps else "not_in_manual",summary="Supplied evaluation history",steps=steps,confidence=0)
        save_turn(store,cid,Turn(question=h.question,answer=answer,cited_pages=h.pages),ActiveContext(pages=h.pages),{"source":"evaluation_history"})
    return cid

def parse_ablations(value: str) -> list[str]:
    """Accept A..E, an individual letter, or a comma-separated subset."""
    names=list("ABCDE") if value.upper()=="A..E" else [x.strip().upper() for x in value.split(",")]
    if not names or any(n not in ABLATIONS for n in names):raise ValueError("Use A..E or a comma-separated subset of A,B,C,D,E")
    return list(dict.fromkeys(names))

def run_evaluation(settings: Settings,dataset: Path,product_id: str,configs: str,output: Path,
                   retrieval_only: bool=False,synthetic: bool=False) -> None:
    """Calibrate each ablation on DEV and report held-out TEST measurements only."""
    records,dataset_hash,manifest=load_dataset(dataset)
    store=Store(settings.db_path)
    product=store.product(product_id)
    source=manifest["source_type"]
    if synthetic!=(source=="synthetic"):raise ValueError("Use --synthetic only for the supplied synthetic dataset")
    if manifest.get("model")!=product.model:raise ValueError("Dataset model does not match selected manual")
    if manifest.get("pdf_sha256") and manifest["pdf_sha256"]!=product.pdf_sha256:raise ValueError("Dataset belongs to a different PDF revision")
    if source=="real" and not manifest.get("pdf_sha256"):raise ValueError("Real datasets must specify pdf_sha256")
    if any(p>product.n_pages for q in records for p in q.gold_pages):raise ValueError("Gold page exceeds manual length")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    skipped={}
    client=None if retrieval_only else AnthropicClient(settings)
    for label in parse_ablations(configs):
        mode,types,use_rerank=ABLATIONS[label]
        if label in {"D","E"} and not any(c.type in {"figure","page_card"} for c in store.chunks(product_id)):
            skipped[label]="No VLM page-card/figure-description chunks. Ingest explicitly with --page-cards and a valid API key."
            continue
        current=settings.model_copy(update={"rerank":use_rerank})
        retriever=Retriever(store,product_id,current,mode=mode,chunk_types=types)
        plans={}
        if retrieval_only:
            plans={q.id:retrieval_plan(q,product.model) for q in records}
        else:
            from manualrag.conversation import rewrite
            for q in [r for r in records if r.split=="dev"]:
                cid=seed_history(store,product_id,q,current)
                _,turns,context=load_conversation(store,product_id,cid,current.history_turns)
                plans[q.id]=rewrite(q.question,product,turns,context,client,current,[])
        dev=[q for q in records if q.split=="dev"]
        scores=[]
        for q in dev:
            hits=retriever.search(plans[q.id])
            # Deterministic scope rejection maps to score -1 for threshold selection.
            decision=gate(plans[q.id],hits,current.model_copy(update={"gate_threshold":0}),product.model)
            scores.append(relevance_score(plans[q.id].standalone_query,hits) if decision.allowed else -1.)
        threshold,f1=calibrate_threshold(scores,[q.answerable for q in dev])
        current=current.model_copy(update={"gate_threshold":threshold,"gate_calibrated":True})
        calibration={"ablation":label,"dataset_sha256":dataset_hash,"pdf_sha256":product.pdf_sha256,
                     "dev_ids":[q.id for q in dev],"threshold":threshold,"dev_abstention_f1":f1}
        atomic_text(output/f"calibration_{label}.json",json.dumps(calibration,indent=2))
        service=ManualAssistant(current,client,mode=mode,chunk_types=types) if client else None
        if service:service.prepare(product_id)
        for q in [r for r in records if r.split=="test"]:
            cid=seed_history(store,product_id,q,current) if service else None
            start=perf_counter()
            if service:
                result=service.ask(product_id,q.question,cid)
                hits=[Hit.model_validate(h) for h in result.trace["retrieved"]]
                decision=GateDecision.model_validate(result.trace["gate"])
            else:
                hits=retriever.search(plans[q.id])
                decision=gate(plans[q.id],hits,current,product.model)
            ranked=[h.chunk.page_no for h in hits if h.role not in {"warning","neighbor"}]
            row={"id":q.id,"ablation":label,"question":q.question,"category":q.category,"answerable":q.answerable,
                 "gold_pages":q.gold_pages,"retrieved_pages":list(dict.fromkeys(ranked)),
                 **retrieval_metrics(ranked,q.gold_pages),"refused":not decision.allowed,"gate_score":decision.score,
                 "citation_precision":None,"citation_recall":None,"faithfulness":None,"gold_point_coverage":None,
                 "component_accuracy":None,"warning_recall":None,"cost_usd":None}
            if service:
                actual_hits=hits
                row["refused"]=result.answer.status in {"not_in_manual","needs_clarification"}
                row["status"]=result.answer.status
                row.update(citation_metrics([v.page_no for v in result.answer.visuals],q.gold_pages))
                usage=list(result.trace["usage"])
                answer_usage=list(usage)
                answer_latency=perf_counter()-start
                judgment=judge_answer(q,result.answer,actual_hits,client,current,usage)
                row["faithfulness"]=sum(c.supported for c in judgment.claims)/len(judgment.claims) if judgment.claims else None
                row["gold_point_coverage"]=sum(judgment.gold_point_coverage)/len(q.gold_points) if q.gold_points else None
                row["component_accuracy"]=float(judgment.component_correct) if q.needs_visual else None
                # Gold-procedure sections define required warnings even if retrieval missed them.
                corpus=store.chunks(product_id)
                gold_sections={c.section_path for c in corpus if c.page_no in q.gold_pages and not c.is_warning}
                required={c.chunk_id for c in corpus if c.is_warning and any(same_section(c.section_path,s) for s in gold_sections)}
                got={w.chunk_id for w in result.answer.warnings}
                row["warning_recall"]=len(required&got)/len(required) if required else None
                if all(u["model"] in current.model_prices for u in answer_usage):
                    row["cost_usd"]=sum(u["input_tokens"]*current.model_prices[u["model"]][0]+u["output_tokens"]*current.model_prices[u["model"]][1] for u in answer_usage)/1e6
                judge_usage=usage[len(answer_usage):]
                row["judge_tokens"]=sum(u["input_tokens"]+u["output_tokens"] for u in judge_usage)
                row["answer_tokens"]=sum(u["input_tokens"]+u["output_tokens"] for u in answer_usage)
                row["judge_cost_usd"]=sum(u["input_tokens"]*current.model_prices[u["model"]][0]+u["output_tokens"]*current.model_prices[u["model"]][1] for u in judge_usage)/1e6 if all(u["model"] in current.model_prices for u in judge_usage) else None
                row["trace"]=result.trace
                row["judgment"]=judgment.model_dump()
            row["latency_seconds"]=answer_latency if service else perf_counter()-start
            rows.append(row)
        logger.info("Ablation %s complete: DEV threshold %.6f",label,threshold)
    metadata={"source_type":source,"mode":"retrieval_only" if retrieval_only else "full","pdf_sha256":product.pdf_sha256,
              "dataset_sha256":dataset_hash,"settings":settings.model_dump(mode="json"),"n_dev":sum(q.split=="dev" for q in records),
              "n_test":sum(q.split=="test" for q in records),"rewrite_backend":"history_concat" if retrieval_only else "FAST_MODEL"}
    write_report(output,rows,metadata,skipped)
