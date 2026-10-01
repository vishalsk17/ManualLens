"""Product-scoped retrieval, exact code lookup, RRF, and mandatory warnings."""
from collections import defaultdict
from typing import Literal
from manualrag.config import Settings
from manualrag.chunking import STEP
from manualrag.index.bm25 import BM25Index
from manualrag.index.store import Store
from manualrag.index.vectors import Embedder, LocalEmbedder, build_vectors
from manualrag.ingest.error_codes import normalize_code
from manualrag.models import Chunk, Hit, QueryPlan
from manualrag.retrieval.rerank import rerank

TEXT_TYPES={"text","table","warning","error_row"}

def rrf(rankings: list[list[tuple[str,float]]],k: int = 60) -> list[tuple[str,float]]:
    """Reciprocal-rank fusion; raw scores never need a shared scale."""
    scores: dict[str,float]=defaultdict(float)
    for ranking in rankings:
        seen=set()
        for rank,(identifier,_) in enumerate(ranking,1):
            if identifier not in seen:scores[identifier]+=1/(k+rank)
            seen.add(identifier)
    return sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]))

def same_section(a: str,b: str) -> bool:
    """Treat a parent section's warnings as applying to its descendants."""
    return a==b or a.startswith(b+" > ") or b.startswith(a+" > ")

def diversify(hits: list[Hit],k: int,per_page: int) -> list[Hit]:
    """Limit repeated pages while preserving relevance order."""
    chosen=[]
    counts: dict[int,int]=defaultdict(int)
    for h in hits:
        if counts[h.chunk.page_no]<per_page:
            chosen.append(h)
            counts[h.chunk.page_no]+=1
        if len(chosen)==k:break
    return chosen

class Retriever:
    """Reusable indexes scoped to one immutable manual."""
    def __init__(self,store: Store,product_id: str,settings: Settings,mode: Literal["bm25","dense","hybrid"]="hybrid",
                 chunk_types: set[str] | None = None,embedder: Embedder | None = None) -> None:
        self.store=store
        self.settings=settings
        self.product=store.product(product_id)
        self.chunks=store.chunks(product_id,chunk_types or TEXT_TYPES)
        self.lookup={c.chunk_id:c for c in self.chunks}
        self.mode=mode
        self.bm25=BM25Index(self.chunks)
        self.embedder=embedder or LocalEmbedder(settings)
        self.vectors=build_vectors(self.chunks,self.product,self.embedder,settings) if self.chunks and mode!="bm25" else None

    def search(self,plan: QueryPlan,enrich: bool = True) -> list[Hit]:
        """Retrieve evidence then add same-section warnings and adjacent steps."""
        query=plan.standalone_query
        exact=[]
        if plan.error_code:
            for e in self.store.errors(self.product.id,normalize_code(plan.error_code)):
                if e.chunk_id in self.lookup:
                    exact.append(Hit(chunk=self.lookup[e.chunk_id],score=1,role="exact"))
                    if e.remedy:query+=" "+e.remedy
        lexical=self.bm25.search(query,self.settings.bm25_k) if self.mode!="dense" else []
        dense=self.vectors.search(self.embedder.encode([query],query=True)[0],self.settings.dense_k) if self.vectors else []
        lex_map=dict(lexical)
        dense_map=dict(dense)
        ranking=rrf([lexical,dense],self.settings.rrf_k)
        hits=[Hit(chunk=self.lookup[i],score=s,lexical_score=lex_map.get(i,0),dense_score=dense_map.get(i,0))
              for i,s in ranking[:self.settings.candidate_k]]
        if self.settings.rerank:hits=rerank(query,hits,self.settings)
        exact_ids={h.chunk.chunk_id for h in exact}
        hits=exact+[h for h in hits if h.chunk.chunk_id not in exact_ids]
        selected=diversify(hits,self.settings.final_k,self.settings.per_page_k)
        if not enrich:return selected
        present={h.chunk.chunk_id for h in selected}
        # Neighbor completion happens before attaching all applicable warnings.
        for hit in list(selected):
            for c in self.chunks:
                if (c.chunk_id not in present and c.type=="text" and same_section(c.section_path,hit.chunk.section_path)
                    and (abs(c.ordinal-hit.chunk.ordinal)<=self.settings.neighbor_chunks or STEP.search(c.text))):
                    selected.append(Hit(chunk=c,score=0,role="neighbor"))
                    present.add(c.chunk_id)
        cited_sections={h.chunk.section_path for h in selected}
        for c in self.store.chunks(self.product.id):
            if c.is_warning and c.chunk_id not in present and any(same_section(c.section_path,s) for s in cited_sections):
                selected.append(Hit(chunk=c,score=0,role="warning"))
                present.add(c.chunk_id)
        if len(selected)>self.settings.max_evidence_chunks:
            raise ValueError("Too much safety context to include. Ask about a more specific procedure.")
        return selected
