"""CSV, Markdown, audit templates, and an explicit Stage 1 qualification gate."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import logging
from manualrag.config import Settings
from manualrag.eval.metrics import summarize
from manualrag.llm.caching import atomic_text

logger=logging.getLogger(__name__)

def format_metric(value: object) -> str:
    """Render unavailable metrics honestly."""
    return "not measured" if value is None else f"{value:.3f}" if isinstance(value,float) else str(value)

def write_report(folder: Path,rows: list[dict],metadata: dict,skipped: dict) -> None:
    """Write per-question evidence, category tables, and five data-driven failure analyses."""
    folder.mkdir(parents=True,exist_ok=True)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r if k not in {"trace","judgment"}))
        with (folder/"metrics.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader()
            for r in rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in keys})
    atomic_text(folder/"results.json",json.dumps({"metadata":metadata,"rows":rows,"skipped":skipped},indent=2,ensure_ascii=False))
    groups=defaultdict(list)
    for row in rows:groups[(row["ablation"],row["category"])].append(row)
    totals={a:summarize([r for r in rows if r["ablation"]==a]) for a in sorted({r["ablation"] for r in rows})}
    atomic_text(folder/"summary.json",json.dumps({"metadata":metadata,"ablations":totals,"skipped":skipped},indent=2))
    lines=["# ManualLens evaluation","",f"Dataset: {metadata['source_type']}. Mode: {metadata['mode']}.",
           "Only held-out TEST rows are reported below; thresholds were selected on DEV.",
           "Synthetic results are engineering smoke checks, not real-manual performance. Small category counts limit conclusions.",
           "Retrieval-only runs measure gate abstention, not generated-answer abstention. LLM faithfulness is unaudited until human review.",
           "Cost is unknown until per-model prices are configured; local embedding/reranker time has no API charge.",
           "Latency includes per-question retrieval and model calls but excludes one-time indexing/model loading.","",
           "| Ablation | Category | n | R@1 | R@3 | R@5 | MRR | Any gold page @5, 95% Wilson CI |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    for (ablation,category),group in sorted(groups.items()):
        stats=summarize(group)
        interval=stats["any_gold_page_hit5_wilson"]
        ci="n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        lines.append(f"| {ablation} | {category} | {len(group)} | "+" | ".join(format_metric(stats[k]) for k in ("recall@1","recall@3","recall@5","mrr"))+f" | {ci} |")
    lines+=["","## Answer quality, abstention, and runtime",""]
    for ablation,stats in totals.items():
        lines.append(f"### {ablation} (n={stats['n']})")
        for key in ("citation_precision","citation_recall","faithfulness","component_accuracy","warning_recall","abstention_precision",
                    "abstention_recall","false_refusal_rate","latency_p50","latency_p95","cost_usd"):
            lines.append(f"- {key}: {format_metric(stats.get(key))}")
    if skipped:
        lines+=["","## Unavailable ablations",""]+[f"- {a}: {reason}" for a,reason in skipped.items()]
    lines+=["","## Five cases for failure analysis",""]
    worst=sorted(rows,key=lambda r:((r.get("recall@5") if r["answerable"] else float(r["refused"])) or 0,r["id"]))[:5]
    for row in worst:
        issues=[]
        if row["answerable"] and (row.get("recall@5") or 0)<1:issues.append("Not all gold pages were retrieved; inspect chunk granularity and query wording.")
        if row["answerable"] and row["refused"]:issues.append("False refusal: inspect the DEV-calibrated gate and source coverage.")
        if not row["answerable"] and not row["refused"]:issues.append("The relevance gate accepted an unanswerable question; generation must still abstain.")
        if row.get("citation_recall") is not None and row["citation_recall"]<1:issues.append("Verified citations omit some gold pages.")
        if row.get("faithfulness") is not None and row["faithfulness"]<1:issues.append("The judge found unsupported claims; inspect the claim-level reasons in results.json.")
        if not issues:issues=["No measured failure in this case. Inspect manually; unmeasured answer quality must not be assumed correct."]
        lines+=[f"### {row['ablation']} / {row['id']}",row["question"],f"Retrieved pages: {row['retrieved_pages']}; gold: {row['gold_pages']}.",*issues,""]
    lines+=["## Human audit","",
            "human_audit.csv is an unfilled worksheet. Review at least 20 distinct answers against the manual, fill faithful as 0 or 1, reviewer, and notes.",
            "The audit command calculates agreement with the LLM judge. No agreement or Stage 1 pass is claimed before that audit."]
    atomic_text(folder/"report.md","\n".join(lines)+"\n")
    audited=[r for r in rows if r["ablation"]=="C" and r.get("faithfulness") is not None][:20]
    audit_path=folder/"human_audit.csv"
    if audit_path.exists() and any(r.get("faithful") in {"0","1"} for r in csv.DictReader(audit_path.open(encoding="utf-8",newline=""))):
        logger.info("Preserved existing human audit worksheet")
        return
    with audit_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["id","ablation","faithful","reviewer","notes"])
        writer.writeheader()
        for r in audited:writer.writerow({"id":r["id"],"ablation":r["ablation"]})
    logger.info("Evaluation report: %s",(folder/"report.md").resolve())

def apply_audit(folder: Path,audit_file: Path,settings: Settings) -> dict:
    """Compute human/LLM agreement; unlock Stage 2 only on a qualified real-manual run."""
    data=json.loads((folder/"results.json").read_text(encoding="utf-8"))
    lookup={(r["id"],r["ablation"]):r for r in data["rows"]}
    audits=[]
    seen=set()
    with audit_file.open(encoding="utf-8",newline="") as stream:
        for a in csv.DictReader(stream):
            key=(a["id"],a["ablation"])
            if key in seen:raise ValueError("Duplicate audit rows")
            if a["ablation"]!="C":raise ValueError("Stage 1 audit must review distinct hybrid text-only (C) answers")
            if a["faithful"] not in {"0","1"} or not a["reviewer"].strip():continue
            if key not in lookup or lookup[key].get("faithfulness") is None:raise ValueError("Audit has no matching judged answer")
            if a["id"] in {k[0] for k in seen}:raise ValueError("Audit requires distinct question IDs")
            seen.add(key)
            audits.append((int(a["faithful"]),int(lookup[key]["faithfulness"]==1)))
    n=len(audits)
    agreement=sum(a==b for a,b in audits)/n if n else None
    result={"n":n,"agreement":agreement,"minimum_met":n>=20,"stage1_pass":False}
    meta=data["metadata"]
    stage1=[r for r in data["rows"] if r["ablation"]=="C"]
    stats=summarize(stage1)
    qualified=(n>=20 and meta["source_type"]=="real" and meta["mode"]=="full"
               and (stats.get("recall@5") or 0)>=settings.stage1_recall5_min
               and (stats.get("abstention_f1") or 0)>=settings.stage1_abstention_f1_min
               and (stats.get("citation_precision") or 0)>=settings.stage1_citation_precision_min
               and (stats.get("warning_recall") or 0)>=settings.stage1_warning_recall_min
               and (stats.get("faithfulness") or 0)>=settings.stage1_faithfulness_min
               and sum(a for a,_ in audits)/n>=settings.stage1_faithfulness_min)
    result["stage1_pass"]=qualified
    atomic_text(folder/"audit_report.json",json.dumps(result,indent=2))
    if qualified:
        atomic_text(settings.data_dir/"stage1_approval.json",json.dumps({"pdf_sha256":meta["pdf_sha256"],
                    "dataset_sha256":meta["dataset_sha256"],"audit":result,"settings":meta["settings"]},indent=2))
    logger.info("Audit: %s",result)
    return result
