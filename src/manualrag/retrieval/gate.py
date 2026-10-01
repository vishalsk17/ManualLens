"""A calibrated relevance gate plus deterministic rejection rules."""
from manualrag.config import Settings
from manualrag.index.bm25 import tokenize
from manualrag.models import GateDecision,Hit,QueryPlan

def relevance_score(query: str,hits: list[Hit]) -> float:
    """Use query-token coverage as a comparable gate feature across ablations."""
    tokens=set(tokenize(query))
    if not tokens or not hits:return 0.0
    return max(len(tokens.intersection(tokenize(h.chunk.text)))/len(tokens) for h in hits if h.role not in {"warning","neighbor"}) if any(h.role not in {"warning","neighbor"} for h in hits) else 0.0

def gate(plan: QueryPlan,hits: list[Hit],settings: Settings,selected_model: str) -> GateDecision:
    """Reject wrong products, out-of-scope requests, unknown codes, or weak evidence."""
    score=relevance_score(plan.standalone_query,hits)
    reason=""
    status="not_in_manual"
    if plan.other_model_mentioned and plan.other_model_mentioned.casefold()!=selected_model.casefold():
        reason=f"This manual covers {selected_model}; the question names a different model."
    elif plan.intent=="out_of_scope":
        reason="The question is outside the scope of this product manual."
    elif not plan.standalone_query.strip():
        reason="Please specify the component, error code, or procedure."
        status="needs_clarification"
    elif plan.error_code and not any(h.role=="exact" for h in hits):
        reason="The requested error code was not found in this manual's verified error-code records."
    elif not hits or score<settings.gate_threshold:
        reason="The retrieved evidence is insufficient for a grounded answer."
    if reason:return GateDecision(allowed=False,status=status,score=score,reason=reason)
    return GateDecision(allowed=True,status="answered",score=score,reason="Evidence passed the relevance threshold; generation must still abstain if insufficient.")

def calibrate_threshold(scores: list[float],answerable: list[bool]) -> tuple[float,float]:
    """Maximize abstention F1 on DEV data only; ties prefer fewer false refusals."""
    if len(scores)!=len(answerable) or not scores or all(answerable) or not any(answerable):
        raise ValueError("Calibration needs both answerable and unanswerable DEV examples")
    candidates=sorted({0.0,1.0,*[s for s in scores if 0<=s<=1],*[min(1.0,s+1e-6) for s in scores if 0<=s<=1]})
    best=(0.0,-1.0,-1.0)
    for threshold in candidates:
        refused=[s<threshold for s in scores]
        tp=sum(r and not a for r,a in zip(refused,answerable))
        fp=sum(r and a for r,a in zip(refused,answerable))
        fn=sum(not r and not a for r,a in zip(refused,answerable))
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        quality=(f1,-fp,-threshold)
        if quality>(best[1],best[2],-best[0]):best=(threshold,f1,-fp)
    return best[0],best[1]
