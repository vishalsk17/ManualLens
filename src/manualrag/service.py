"""The UI and CLI use the same typed application service."""
from pathlib import Path
from time import perf_counter
from manualrag.answer.generate import generate_answer
from manualrag.answer.verify import verify_answer,STATUS_TEXT
from manualrag.config import Settings
from manualrag.conversation import load_conversation,rewrite,save_turn
from manualrag.index.store import Store
from manualrag.index.vectors import Embedder
from manualrag.llm.client import LLMClient
from manualrag.models import ActiveContext,Answer,AskResult,Turn
from manualrag.retrieval.gate import gate
from manualrag.retrieval.hybrid import Retriever
from manualrag.visuals import visual_payloads

class ManualAssistant:
    """Orchestrate re-retrieval, abstention, generation, verification, and trace persistence."""
    def __init__(self,settings: Settings,client: LLMClient,mode: str="hybrid",
                 chunk_types: set[str] | None=None,embedder: Embedder | None=None) -> None:
        self.settings=settings
        self.client=client
        self.mode=mode
        self.chunk_types=chunk_types
        self.embedder=embedder
        self.store=Store(settings.db_path)
        self._retrievers: dict[str,Retriever]={}

    def prepare(self,product_id: str) -> Retriever:
        """Warm a product index outside timed evaluation calls."""
        if product_id not in self._retrievers:
            self._retrievers[product_id]=Retriever(self.store,product_id,self.settings,self.mode,self.chunk_types,self.embedder)
        return self._retrievers[product_id]

    def ask(self,product_id: str,question: str,conversation_id: str | None=None) -> AskResult:
        """Answer against a selected manual; failed model calls never become product advice."""
        start=perf_counter()
        product=self.store.product(product_id)
        conversation_id,turns,context=load_conversation(self.store,product_id,conversation_id,self.settings.history_turns)
        usage: list[dict]=[]
        plan=rewrite(question,product,turns,context,self.client,self.settings,usage)
        hits=self.prepare(product_id).search(plan)
        decision=gate(plan,hits,self.settings,product.model)
        raw=None
        if decision.allowed:
            images=[]
            if plan.needs_visual:
                pages=list(dict.fromkeys(h.chunk.page_no for h in hits))[:self.settings.visual_pages]
                images=[Path(self.store.page(product_id,p).png_path) for p in pages]
            raw=generate_answer(plan,hits,product,self.client,self.settings,images,usage)
        else:
            raw=Answer(status=decision.status,summary=STATUS_TEXT[decision.status],confidence=0,
                       missing_info=[decision.reason,f"Searched this manual for: {plan.standalone_query}"])
        verified,issues=verify_answer(raw,hits,product_id)
        if not decision.allowed:
            verified.missing_info.append(decision.reason)
        if verified.status=="not_in_manual" and not any(m.startswith("Searched this manual") for m in verified.missing_info):
            verified.missing_info.append(f"Searched this manual for: {plan.standalone_query}")
        visuals=visual_payloads(verified,hits,self.store,product_id,self.settings.cache_dir/"overlays")
        pages=sorted({v.page_no for v in verified.visuals})
        new_context=ActiveContext(pages=pages,component=plan.component or context.component,
                                  error_code=plan.error_code or context.error_code)
        trace={"product_id":product_id,"pdf_sha256":product.pdf_sha256,"rewritten_query":plan.standalone_query,
               "plan":plan.model_dump(),"retrieved":[h.model_dump() for h in hits],
               "gate":decision.model_dump(),"gate_calibrated":self.settings.gate_calibrated,
               "raw_answer":raw.model_dump(),"verified_answer":verified.model_dump(),"verification_issues":issues,
               "latency_seconds":perf_counter()-start,"usage":usage,"prompt_version":self.settings.prompt_version}
        save_turn(self.store,conversation_id,Turn(question=question,answer=verified,cited_pages=pages),new_context,trace)
        return AskResult(conversation_id=conversation_id,answer=verified,visuals=visuals,trace=trace)
