"""Local embeddings and a swappable NumPy/FAISS vector-store interface."""
from pathlib import Path
from typing import Protocol
from functools import lru_cache
import numpy as np
from manualrag.config import Settings
from manualrag.llm.caching import cache_key
from manualrag.models import Chunk, Product

class Embedder(Protocol):
    """Embedding adapter shared by production inference and deterministic test doubles."""
    def encode(self,texts: list[str],query: bool = False) -> np.ndarray: ...

@lru_cache(maxsize=3)
def _load_model(name: str,cache_folder: str,revision: str | None) -> object:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name,cache_folder=cache_folder,revision=revision)

class LocalEmbedder:
    """Run the configured BGE model locally, without substituting fake embeddings."""
    def __init__(self,settings: Settings) -> None:
        self.settings=settings

    def encode(self,texts: list[str],query: bool = False) -> np.ndarray:
        """Generate normalized dense vectors with the BGE retrieval query instruction."""
        model=_load_model(self.settings.embedding_model,str(self.settings.cache_dir/"models"),self.settings.embedding_revision)
        if query and "bge-" in self.settings.embedding_model:
            texts=["Represent this sentence for searching relevant passages: "+text for text in texts]
        return np.asarray(model.encode(texts,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)

class VectorStore(Protocol):
    """Minimal vector backend contract."""
    def search(self,query: np.ndarray,k: int) -> list[tuple[str,float]]: ...

class NumpyVectorStore:
    """Exact cosine search, sufficient for one manual."""
    def __init__(self,ids: list[str],matrix: np.ndarray) -> None:
        self.ids=ids
        norms=np.linalg.norm(matrix,axis=1,keepdims=True)
        self.matrix=(matrix/np.maximum(norms,1e-12)).astype(np.float32)

    def search(self,query: np.ndarray,k: int) -> list[tuple[str,float]]:
        """Return deterministic nearest neighbors."""
        q=query.reshape(-1)
        q=q/max(float(np.linalg.norm(q)),1e-12)
        scores=self.matrix@q
        order=np.argsort(-scores,kind="stable")[:k]
        return [(self.ids[i],float(scores[i])) for i in order]

class FaissVectorStore(NumpyVectorStore):
    """Optional exact inner-product FAISS backend using normalized vectors."""
    def __init__(self,ids: list[str],matrix: np.ndarray) -> None:
        super().__init__(ids,matrix)
        import faiss
        self.index=faiss.IndexFlatIP(self.matrix.shape[1])
        self.index.add(self.matrix)

    def search(self,query: np.ndarray,k: int) -> list[tuple[str,float]]:
        """Return cosine neighbors from FAISS."""
        q=query.reshape(1,-1).astype(np.float32)
        q/=max(float(np.linalg.norm(q)),1e-12)
        scores,indices=self.index.search(q,min(k,len(self.ids)))
        return [(self.ids[i],float(s)) for i,s in zip(indices[0],scores[0]) if i>=0]

def build_vectors(chunks: list[Chunk],product: Product,embedder: Embedder,settings: Settings) -> VectorStore:
    """Cache each embedding by PDF, page, version, model, and contextual text."""
    folder=settings.cache_dir/"embeddings"
    folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    missing=[]
    positions=[]
    vectors: list[np.ndarray | None]=[None]*len(chunks)
    for i,c in enumerate(chunks):
        key=cache_key(product.pdf_sha256,c.page_no,settings.prompt_version,model=settings.embedding_model,revision=settings.embedding_revision,
                      text=c.embed_text,adapter=type(embedder).__name__)
        path=folder/f"{key}.npy"
        paths.append(path)
        if path.exists():vectors[i]=np.load(path,allow_pickle=False)
        else:
            missing.append(c.embed_text)
            positions.append(i)
    if missing:
        values=embedder.encode(missing)
        for i,vector in zip(positions,values):
            vectors[i]=vector
            # np.save never serializes Python objects.
            from uuid import uuid4
            temp=paths[i].with_suffix("."+uuid4().hex+".tmp")
            try:
                with temp.open("wb") as stream:np.save(stream,vector,allow_pickle=False)
                temp.replace(paths[i])
            finally:
                temp.unlink(missing_ok=True)
    matrix=np.stack(vectors).astype(np.float32)
    cls=FaissVectorStore if settings.vector_backend=="faiss" else NumpyVectorStore
    return cls([c.chunk_id for c in chunks],matrix)
