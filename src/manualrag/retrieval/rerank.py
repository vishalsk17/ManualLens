"""Optional local cross-encoder, disabled by default."""
from functools import lru_cache
from manualrag.config import Settings
from manualrag.models import Hit

@lru_cache(maxsize=2)
def _load(name: str) -> object:
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)

def rerank(query: str,hits: list[Hit],settings: Settings) -> list[Hit]:
    """Reorder candidates by cross-encoder relevance without discarding provenance."""
    if not hits:return []
    scores=_load(settings.reranker_model).predict([(query,h.chunk.text) for h in hits])
    return [h.model_copy(update={"score":float(score)}) for h,score in sorted(zip(hits,scores),key=lambda pair:-pair[1])]
