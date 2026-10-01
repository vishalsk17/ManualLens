"""Regression tests for grounding, safety, history, and evidence geometry."""
from pathlib import Path
import json
import pytest
from manualrag.config import Settings
from manualrag.chunking import chunk_blocks,split_procedure
from manualrag.models import Answer,Block,Product,QueryPlan,Step
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.index.store import Store
from manualrag.llm.client import Completion,structured_call
from manualrag.service import ManualAssistant

class ManualClient:
    """Deterministic provider double; no answers are invented outside supplied chunks."""
    def __init__(self) -> None:
        self.calls: list[dict]=[]

    def complete(self,**kwargs: object) -> Completion:
        self.calls.append(kwargs)
        payload=kwargs["payload"]
        if "latest_question" in payload:
            followup=payload["latest_question"]=="Which cover?"
            if followup:
                assert payload["history"]
                assert payload["active_context"]["component"]=="rear access cover"
            plan=QueryPlan(standalone_query="Which rear access cover opens for a paper jam?" if followup else "How do I clear a paper jam?",
                           intent="locate_part" if followup else "how_to",component="rear access cover",needs_visual=followup)
            return Completion(plan.model_dump_json(),10,10,kwargs["model"])
        evidence=payload["evidence"]
        support=next(c for c in evidence if "1. Press Power to turn off" in c["text"])
        answer=Answer(status="answered",summary="This uncited summary must be replaced.",confidence=.8,
                      steps=[Step(n=1,text="Turn off and unplug the printer before opening the rear access cover.",chunk_ids=[support["chunk_id"]])])
        return Completion(answer.model_dump_json(),10,10,kwargs["model"])

def test_answer_followup_warnings_trace_and_visuals(manual: Path,settings: Settings) -> None:
    product=ingest_pdf(manual,"Lab","ML-100",settings)
    client=ManualClient()
    assistant=ManualAssistant(settings,client,mode="bm25")
    first=assistant.ask(product.id,"How do I clear a paper jam?")
    second=assistant.ask(product.id,"Which cover?",first.conversation_id)
    assert first.answer.status=="answered"
    assert len(first.answer.warnings)>=2
    assert any("Do not touch the rollers" in w.text for w in first.answer.warnings)
    assert all(w.text==next(h["chunk"]["text"] for h in first.trace["retrieved"] if h["chunk"]["chunk_id"]==w.chunk_id) for w in first.answer.warnings)
    assert any(v.page_no==4 and v.rectangles for v in first.visuals)
    assert second.conversation_id==first.conversation_id
    assert second.trace["rewritten_query"].startswith("Which rear access")
    assert len(client.calls)==4
    assert 1<=len(client.calls[-1]["images"])<=2
    with assistant.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM turns WHERE conversation_id=?",(first.conversation_id,)).fetchone()[0]==2
        assert db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]==2

def test_mixed_warning_block_and_long_prose() -> None:
    product=Product(id="p",name="Printer",model="M1",pdf_sha256="x",n_pages=1)
    blocks=[Block(page_no=1,text="Before opening\nWARNING: Unplug the device.",bbox=(1,1,50,50),font_size=11,section_path="Jam")]
    chunks=chunk_blocks(blocks,product)
    assert chunks[0].is_warning
    parts=split_procedure("word "*1000,350)
    assert len(parts)>=4 and sum(len(p.split()) for p in parts)==1000

def test_schema_retry_can_recover() -> None:
    class Recovering:
        calls=0
        def complete(self,**kwargs: object) -> Completion:
            self.calls+=1
            return Completion("invalid" if self.calls==1 else Answer(status="not_in_manual",summary="Unknown",confidence=0).model_dump_json())
    client=Recovering()
    answer=structured_call(client,Answer,system="x",payload={},model="test")
    assert answer.status=="not_in_manual" and client.calls==2

def test_reingest_cached_cards_makes_zero_new_calls(manual: Path,settings: Settings) -> None:
    class Cards:
        calls=0
        def complete(self,**kwargs: object) -> Completion:
            self.calls+=1
            return Completion('{"summary":"Visible text only","figures":[],"labels":{},"warning_texts":[],"uncertainty_notes":[]}')
    client=Cards()
    cfg=settings.model_copy(update={"enable_page_cards":True})
    p=ingest_pdf(manual,"Lab","ML-100",cfg,llm=client)
    assert client.calls==6
    ingest_pdf(manual,"Lab","ML-100",cfg,llm=client)
    assert client.calls==6
    assert sum(c.type=="page_card" for c in Store(cfg.db_path).chunks(p.id))==6
