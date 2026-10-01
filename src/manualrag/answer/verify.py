"""Fail-closed citation verification and automatic verbatim warning restoration."""
from manualrag.models import Answer,Hit,Step,Warning,VisualRequest
from manualrag.retrieval.hybrid import same_section

STATUS_TEXT={
    "answered":"The manual supports the following answer.",
    "partial":"The manual supports part of the answer.",
    "not_in_manual":"I could not find sufficient information in this manual. Contact the manufacturer for help.",
    "needs_clarification":"Please clarify the component, error code, or procedure you mean."
}

def verify_answer(raw: Answer,hits: list[Hit],product_id: str) -> tuple[Answer,list[str]]:
    """Reject invalid citations and supply required warnings directly from source chunks."""
    lookup={h.chunk.chunk_id:h.chunk for h in hits if h.chunk.product_id==product_id}
    issues=[]
    steps=[]
    if raw.status in {"answered","partial"}:
        for step in raw.steps:
            if not step.chunk_ids or any(i not in lookup for i in step.chunk_ids):
                issues.append(f"Dropped step {step.n}: citation not in retrieved manual evidence")
                continue
            steps.append(step.model_copy(update={"n":len(steps)+1,"chunk_ids":list(dict.fromkeys(step.chunk_ids))}))
    status=raw.status
    if status in {"answered","partial"} and not steps:
        status="not_in_manual"
        issues.append("No valid cited factual steps survived verification")
    elif issues and status=="answered":status="partial"
    cited={i for s in steps for i in s.chunk_ids}
    sections={lookup[i].section_path for i in cited}
    warnings=[Warning(text=c.text,chunk_id=c.chunk_id) for c in lookup.values()
              if c.is_warning and any(same_section(c.section_path,s) for s in sections)]
    raw_warnings={w.chunk_id:w.text for w in raw.warnings}
    for w in warnings:
        if raw_warnings.get(w.chunk_id)!=w.text:
            issues.append(f"Restored verbatim safety text: {w.chunk_id}")
    pages=sorted({lookup[i].page_no for i in cited}|{lookup[w.chunk_id].page_no for w in warnings})
    # No LLM-generated visual rationale or page number can introduce new evidence.
    visuals=[VisualRequest(page_no=p,kind="page",reason="Cited manual evidence") for p in pages]
    missing={
        "answered":[],
        "partial":["Some requested details could not be supported by the retrieved manual passages."],
        "not_in_manual":["The retrieved passages did not support an answer."],
        "needs_clarification":["Specify the component, error code, or procedure you want help with."]
    }[status]
    return Answer(status=status,summary=STATUS_TEXT[status],steps=steps,warnings=warnings,visuals=visuals,
                  confidence=raw.confidence if steps else 0,missing_info=missing),issues
