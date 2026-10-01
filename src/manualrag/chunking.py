"""Structure-aware chunks with PDF-derived geometry and contextual prefixes."""
from hashlib import sha256
import re
from manualrag.models import Block, Chunk, Product, BBox

WARNING = re.compile(r"^\s*(?:WARNING|CAUTION|NOTE|IMPORTANT)\b", re.I | re.M)
STEP = re.compile(r"(?m)^\s*\d+[.)]\s+")

def union_bbox(boxes: list[BBox]) -> BBox:
    """Return the enclosing PDF rectangle for source blocks."""
    return min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)

def make_chunk(product: Product, page_no: int, section: str, text: str, ordinal: int,
               kind: str = "text", bbox: BBox | None = None, bbox_source: str = "pdf",
               figure_id: str | None = None) -> Chunk:
    """Build a stable content-addressed chunk with contextual embedding text."""
    identifier = sha256(f"{product.id}|{page_no}|{kind}|{section}|{ordinal}|{text}".encode("utf-8")).hexdigest()[:24]
    return Chunk(chunk_id=identifier,product_id=product.id,page_no=page_no,type=kind,section_path=section,
                 text=text,embed_text=f"{product.name} {product.model} › {section} › p.{page_no}\n{text}",
                 bbox=bbox,bbox_source=bbox_source if bbox else "none",is_warning=kind=="warning",
                 figure_id=figure_id,ordinal=ordinal)

def split_procedure(text: str, max_tokens: int) -> list[str]:
    """Keep a short numbered procedure intact; split long ones only at step boundaries."""
    if not STEP.search(text):
        words=text.split()
        limit=max(1,int(max_tokens/1.35))
        return [" ".join(words[i:i+limit]) for i in range(0,len(words),limit)] if len(words)>limit else [text]
    parts = re.split(r"(?m)(?=^\s*\d+[.)]\s+)", text)
    result: list[str] = []
    current = ""
    for part in parts:
        # A single oversized step is preserved to avoid losing procedural meaning.
        if current and len((current+part).split()) * 1.35 > max_tokens:
            result.append(current.strip())
            current = ""
        current += part
    if current.strip():
        result.append(current.strip())
    return result

def chunk_blocks(blocks: list[Block], product: Product, max_tokens: int = 350) -> list[Chunk]:
    """Group adjacent numbered steps within a page/section and keep warnings separate."""
    groups: list[list[Block]] = []
    for b in blocks:
        if groups and STEP.match(b.text) and not WARNING.search(b.text):
            previous = groups[-1]
            if (previous[-1].page_no,previous[-1].section_path)==(b.page_no,b.section_path) and STEP.search(previous[0].text):
                previous.append(b)
                continue
        groups.append([b])
    chunks: list[Chunk] = []
    for group in groups:
        first = group[0]
        text = "\n".join(b.text for b in group)
        kind = "warning" if WARNING.search(text) else "text"
        pieces = [text] if kind=="warning" else split_procedure(text,max_tokens)
        for piece in pieces:
            chunks.append(make_chunk(product,first.page_no,first.section_path,piece,len(chunks),
                                     kind,union_bbox([b.bbox for b in group])))
    return chunks
