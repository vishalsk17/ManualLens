"""Extract text spans, exact block geometry, and document section context."""
from collections import Counter
from statistics import median
import pymupdf
from manualrag.models import Block

def extract_blocks(doc: pymupdf.Document, heading_ratio: float = 1.15) -> list[Block]:
    """Prefer the PDF outline and fall back to larger or bold short text."""
    toc = doc.get_toc()
    output: list[Block] = []
    section_stack: list[str] = []
    for page_index, page in enumerate(doc):
        page.set_rotation(0)
        for level, title, page_no in toc:
            if page_no == page_index + 1:
                section_stack = section_stack[:level-1] + [title]
        raw = page.get_text("dict", sort=True)["blocks"]
        sizes = [s["size"] for b in raw if b["type"] == 0 for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        body = Counter(round(s,1) for s in sizes).most_common(1)[0][0] if sizes else 11
        section = " > ".join(section_stack) or "Manual"
        for b in raw:
            if b["type"] != 0:
                continue
            text = "\n".join("".join(s["text"] for s in line["spans"]).strip() for line in b["lines"]).strip()
            if not text:
                continue
            spans = [s for line in b["lines"] for s in line["spans"]]
            size = max(s["size"] for s in spans)
            bold = any(s["flags"] & 16 for s in spans)
            is_heading = len(text.split()) <= 14 and (size > body * heading_ratio or (bold and size >= body))
            if is_heading and not text.lower().startswith(("warning", "caution", "note")):
                if toc:
                    if text.casefold() not in [s.casefold() for s in section_stack]:
                        section = " > ".join(section_stack + [text])
                else:
                    section = text
            output.append(Block(page_no=page_index+1,text=text,bbox=tuple(b["bbox"]),font_size=size,bold=bold,section_path=section))
    return output
