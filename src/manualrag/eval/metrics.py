"""Transparent metrics with explicit denominators."""
from math import sqrt
from statistics import mean
import numpy as np

def wilson(successes: int,n: int,z: float=1.959963984540054) -> tuple[float,float] | None:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if n==0:return None
    p=successes/n
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0.,center-half),min(1.,center+half)

def retrieval_metrics(pages: list[int],gold: list[int]) -> dict[str,float | None]:
    """Compute distinct-page recall and reciprocal rank."""
    ranked=list(dict.fromkeys(pages))
    if not gold:return {**{f"recall@{k}":None for k in (1,3,5)},"mrr":None}
    result={f"recall@{k}":len(set(ranked[:k])&set(gold))/len(set(gold)) for k in (1,3,5)}
    result["mrr"]=next((1/(i+1) for i,p in enumerate(ranked) if p in gold),0.)
    return result

def citation_metrics(cited: list[int],gold: list[int]) -> dict[str,float | None]:
    """Compare verified cited pages to human gold pages, separately from ID validity."""
    c,g=set(cited),set(gold)
    return {"citation_precision":len(c&g)/len(c) if c else None,
            "citation_recall":len(c&g)/len(g) if g else None}

def abstention_metrics(answerable: list[bool],refused: list[bool]) -> dict[str,float | int | None]:
    """Treat abstaining on an unanswerable question as the positive class."""
    tp=sum(not a and r for a,r in zip(answerable,refused))
    fp=sum(a and r for a,r in zip(answerable,refused))
    fn=sum(not a and not r for a,r in zip(answerable,refused))
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    return {"abstention_precision":precision,"abstention_recall":recall,
            "abstention_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "false_refusal_rate":fp/sum(answerable) if sum(answerable) else None,
            "abstention_tp":tp,"abstention_fp":fp,"abstention_fn":fn}

def summarize(rows: list[dict]) -> dict:
    """Aggregate available measurements without turning missing results into zeros."""
    metric_names=["recall@1","recall@3","recall@5","mrr","citation_precision","citation_recall",
                  "faithfulness","gold_point_coverage","component_accuracy","warning_recall","cost_usd"]
    result={"n":len(rows),"n_answerable":sum(r["answerable"] for r in rows)}
    for metric in metric_names:
        values=[r[metric] for r in rows if r.get(metric) is not None]
        result[metric]=mean(values) if values else None
        result[f"{metric}_n"]=len(values)
    answerable=[r for r in rows if r["answerable"]]
    result["any_gold_page_hit5_wilson"]=wilson(sum((r.get("recall@5") or 0)>0 for r in answerable),len(answerable))
    if rows:
        result.update(abstention_metrics([r["answerable"] for r in rows],[r["refused"] for r in rows]))
        result["latency_p50"]=float(np.percentile([r["latency_seconds"] for r in rows],50))
        result["latency_p95"]=float(np.percentile([r["latency_seconds"] for r in rows],95))
    return result
