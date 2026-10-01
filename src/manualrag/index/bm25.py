"""BM25 with identifiers preserved across punctuation variants."""
import re
from rank_bm25 import BM25Okapi
from manualrag.ingest.error_codes import CODE_PATTERN, normalize_code
from manualrag.models import Chunk

STOPWORDS=set("a an the is are was were to of in on for and or with how what where when why do does i it this that can my me please tell about".split())

def tokenize(text: str) -> list[str]:
    """Keep error codes as complete tokens, including F-12 and e 04."""
    text=CODE_PATTERN.sub(lambda m:normalize_code(m.group()),text)
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*",text.lower()) if t not in STOPWORDS]

class BM25Index:
    """A lexical index over one already product-filtered corpus."""
    def __init__(self,chunks: list[Chunk]) -> None:
        self.chunks=chunks
        self.index=BM25Okapi([tokenize(c.embed_text) or ["__empty__"] for c in chunks]) if chunks else None

    def search(self,query: str,k: int) -> list[tuple[str,float]]:
        """Return positive-scoring candidates with stable tie-breaking."""
        if self.index is None:return []
        scores=self.index.get_scores(tokenize(query))
        return sorted(((c.chunk_id,float(score)) for c,score in zip(self.chunks,scores) if score>0),
                      key=lambda pair:(-pair[1],pair[0]))[:k]
