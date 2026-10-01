"""Exercise the live-evaluation wiring with a clearly labelled provider double."""
from pathlib import Path
import json
import csv
import pytest
from manualrag.config import Settings
from manualrag.eval.runner import run_evaluation
from manualrag.eval.report import apply_audit
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.llm.client import Completion
from manualrag.models import Answer,QueryPlan,Step

class EvalClient:
    """Fake provider for orchestration tests only, never for reported benchmark scores."""
    def complete(self,**kwargs: object) -> Completion:
        payload=kwargs["payload"]
        if "latest_question" in payload:
            question=payload["latest_question"]
            code="E99" if "E99" in question else "F99" if "F99" in question else None
            return Completion(QueryPlan(standalone_query=question,intent="error_code" if code else "how_to",error_code=code).model_dump_json(),10,10,kwargs["model"])
        if "manual_evidence" in payload:
            answer=payload["answer"]
            result={"claims":[{"step_n":s["n"],"claim":s["text"],"supported":True,"reason":"Synthetic test double."} for s in answer["steps"]],
                    "gold_point_coverage":[bool(answer["steps"]) for _ in payload["gold_points"]],"component_correct":None}
            return Completion(json.dumps(result),5,5,kwargs["model"])
        support=next(c for c in payload["evidence"] if "1. Press Power to turn off" in c["text"])
        answer=Answer(status="answered",summary="Neutral",confidence=.8,
                      steps=[Step(n=1,text="Turn off and unplug the printer.",chunk_ids=[support["chunk_id"]])])
        return Completion(answer.model_dump_json(),20,20,kwargs["model"])

def test_full_runner_and_separate_judge_cost(manual: Path,settings: Settings,tmp_path: Path,monkeypatch) -> None:
    product=ingest_pdf(manual,"Lab","ML-100",settings)
    dataset=tmp_path/"questions.jsonl"
    records=[
        dict(id="d1",question="How do I clear a paper jam?",category="jams",answerable=True,gold_pages=[4],gold_points=["Unplug"],split="dev"),
        dict(id="d2",question="E99",category="unanswerable",answerable=False,gold_pages=[],gold_points=[],split="dev"),
        dict(id="t1",question="Clear jammed paper safely",category="jams",answerable=True,gold_pages=[4],gold_points=["Unplug"],split="test"),
        dict(id="t2",question="F99",category="unanswerable",answerable=False,gold_pages=[],gold_points=[],split="test")]
    dataset.write_text("\n".join(json.dumps(r) for r in records),encoding="utf-8")
    dataset.with_suffix(".manifest.json").write_text(json.dumps({"source_type":"synthetic","model":"ML-100"}),encoding="utf-8")
    monkeypatch.setattr("manualrag.eval.runner.AnthropicClient",lambda config:EvalClient())
    settings=settings.model_copy(update={"model_prices":{settings.fast_model:(1,5),settings.answer_model:(2,10)}})
    output=tmp_path/"full"
    run_evaluation(settings,dataset,product.id,"A",output,False,True)
    result=json.loads((output/"results.json").read_text(encoding="utf-8"))
    answered=next(r for r in result["rows"] if r["id"]=="t1")
    assert answered["faithfulness"]==1 and answered["warning_recall"]==1
    assert answered["answer_tokens"]==60 and answered["judge_tokens"]==10
    assert answered["cost_usd"]==pytest.approx(.0003)
    assert answered["judge_cost_usd"]==pytest.approx(.00003)

def test_audit_rejects_cross_ablation_reviews(tmp_path: Path,settings: Settings) -> None:
    (tmp_path/"results.json").write_text(json.dumps({"metadata":{"source_type":"synthetic","mode":"full"},"rows":[]}),encoding="utf-8")
    audit=tmp_path/"audit.csv"
    audit.write_text("id,ablation,faithful,reviewer,notes\nx,A,1,Tester,checked\n",encoding="utf-8")
    with pytest.raises(ValueError,match="hybrid text-only"):
        apply_audit(tmp_path,audit,settings)
