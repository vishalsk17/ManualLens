"""Normalize codes while preserving verbatim source provenance."""
import re
from manualrag.models import Chunk, ErrorCode, Page, Product

CODE_PATTERN=re.compile(r"\b[A-Za-z]{1,3}(?:[- ]?\d){2,4}\b")

def normalize_code(code: str) -> str:
    """Make E04, E-04, and e 04 share one exact lookup key."""
    return re.sub(r"[\s-]+","",code).upper()

def validate_error_row(row: ErrorCode, page_text: str, product_id: str) -> bool:
    """Reject rows whose raw code is absent verbatim from the claimed page."""
    return bool(row.code_raw and re.search(r"(?<!\w)"+re.escape(row.code_raw)+r"(?!\w)",page_text) and row.product_id==product_id
                and normalize_code(row.code_raw)==row.code_norm)

def extract_error_codes(product: Product, pages: list[Page], chunks: list[Chunk]) -> list[ErrorCode]:
    """Extract row-level table codes; never infer meanings from outside the PDF."""
    page_text={p.page_no:p.text for p in pages}
    result=[]
    seen=set()
    for c in chunks:
        if c.type not in {"table","error_row"} or "\n" in c.text or "code:" not in c.text.lower():
            continue
        fields={part.split(":",1)[0].strip().lower():part.split(":",1)[1].strip()
                for part in c.text.split("|") if ":" in part}
        raw=fields.get("code","")
        if not CODE_PATTERN.fullmatch(raw):
            continue
        row=ErrorCode(product_id=product.id,code_raw=raw,code_norm=normalize_code(raw),
                      meaning=fields.get("meaning",""),cause=fields.get("cause",""),remedy=fields.get("remedy",""),
                      page_no=c.page_no,chunk_id=c.chunk_id)
        key=(row.code_norm,row.chunk_id)
        if validate_error_row(row,page_text[c.page_no],product.id) and key not in seen:
            result.append(row)
            seen.add(key)
            c.type="error_row"
    # Non-table lines, e.g. 'E04: Paper jam. ...', preserve the complete line as meaning.
    for c in chunks:
        if c.type!="text":continue
        for line in c.text.splitlines():
            match=re.match(r"^\s*("+CODE_PATTERN.pattern[2:-2]+r")\s*[:=]\s*(.+)$",line)
            if not match:continue
            raw=match.group(1)
            row=ErrorCode(product_id=product.id,code_raw=raw,code_norm=normalize_code(raw),meaning=match.group(2),
                          page_no=c.page_no,chunk_id=c.chunk_id)
            if validate_error_row(row,page_text[c.page_no],product.id):
                result.append(row)
    return result
