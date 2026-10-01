from pathlib import Path
import numpy as np
from manualrag.config import Settings
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.index.bm25 import tokenize
from manualrag.index.store import Store
from manualrag.index.vectors import NumpyVectorStore,build_vectors
from manualrag.models import QueryPlan
from manualrag.retrieval.hybrid import Retriever,rrf

def test_tokens_and_rrf() -> None:
    assert tokenize("E-04 or e 04 F-12")==["e04","e04","f12"]
    ranked=rrf([[("a",10),("b",2)],[("b",99),("c",8)]])
    assert ranked[0][0]=="b"
    assert abs(ranked[0][1]-(1/62+1/61))<1e-9

def test_numpy_cosine() -> None:
    store=NumpyVectorStore(["a","b"],np.array([[1,0],[0,5]],dtype=np.float32))
    assert store.search(np.array([0,2]),1)==[("b",1.0)]

def test_retrieval_recall_and_warning_attachment(manual: Path,settings: Settings) -> None:
    p=ingest_pdf(manual,"Lab printer","ML-100",settings)
    r=Retriever(Store(settings.db_path),p.id,settings,mode="bm25")
    questions=[("blue shipping tape",1),("Ready indicator control panel",2),("F-12",3),("jammed paper rear cover",4),("faint printing cleaning cycle",5)]
    found=0
    for q,gold in questions:
        hits=r.search(QueryPlan(standalone_query=q,intent="how_to"),enrich=False)
        pages=list(dict.fromkeys(h.chunk.page_no for h in hits))[:5]
        found+=gold in pages
    assert found/len(questions)==1.0
    hits=r.search(QueryPlan(standalone_query="Clear paper jam",intent="how_to"))
    assert any(h.chunk.is_warning and "unplug" in h.chunk.text for h in hits)
    exact=r.search(QueryPlan(standalone_query="E-04",intent="error_code",error_code="E-04"))
    assert exact[0].role=="exact" and exact[0].chunk.page_no==3
    assert all(h.chunk.product_id==p.id for h in exact)

def test_embedding_cache_zero_repeat_calls(manual: Path,settings: Settings) -> None:
    p=ingest_pdf(manual,"Lab printer","ML-100",settings)
    chunks=Store(settings.db_path).chunks(p.id)[:3]
    class Counter:
        calls=0
        def encode(self,texts: list[str],query: bool=False) -> np.ndarray:
            self.calls+=1
            return np.ones((len(texts),4),dtype=np.float32)
    e=Counter()
    build_vectors(chunks,p,e,settings)
    build_vectors(chunks,p,e,settings)
    assert e.calls==1
