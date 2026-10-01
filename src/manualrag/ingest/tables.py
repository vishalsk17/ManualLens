"""Table extraction retaining exact row geometry."""
import pymupdf
from manualrag.chunking import make_chunk
from manualrag.models import Chunk, Product

def extract_tables(page: pymupdf.Page, product: Product, section: str, ordinal: int) -> list[Chunk]:
    """Create table Markdown plus individually retrievable rows."""
    result: list[Chunk] = []
    for table in page.find_tables().tables:
        rows = [[(cell or "").replace("\n"," ").strip() for cell in row] for row in table.extract()]
        if not rows:
            continue
        header = rows[0]
        markdown = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
        markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
        result.append(make_chunk(product,page.number+1,section,markdown,ordinal+len(result),"table",tuple(table.bbox)))
        for index,row in enumerate(rows[1:],1):
            text = " | ".join(f"{h}: {v}" for h,v in zip(header,row))
            result.append(make_chunk(product,page.number+1,section,text,ordinal+len(result),"table",tuple(table.rows[index].bbox)))
    return result
