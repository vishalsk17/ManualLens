"""Atomic ingestion of user-provided manuals."""
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
import logging
import shutil
import pymupdf

from manualrag.chunking import chunk_blocks, make_chunk
from manualrag.ingest.figures import extract_figures
from manualrag.ingest.error_codes import extract_error_codes
from manualrag.ingest.vlm_cards import page_card
from manualrag.llm.client import LLMClient
from manualrag.config import Settings
from manualrag.index.store import Store
from manualrag.ingest.parse import extract_blocks
from manualrag.ingest.render import render_page
from manualrag.ingest.tables import extract_tables
from manualrag.models import Product

logger = logging.getLogger(__name__)

def ingest_pdf(path: Path, name: str, model: str, settings: Settings,
               progress: Callable[[float,str],None] | None = None, llm: LLMClient | None = None) -> Product:
    """Render and index a PDF, committing only after every page succeeds."""
    if not name.strip() or not model.strip():
        raise ValueError("Product name and model are required")
    if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        raise ValueError("PDF exceeds configured size limit")
    digest = sha256(path.read_bytes()).hexdigest()
    product_id = sha256(f"{digest}|{name.strip()}|{model.strip()}".encode("utf-8")).hexdigest()[:20]
    folder = settings.cache_dir / "manuals" / digest
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "manual.pdf"
    if path.resolve() != target.resolve():
        shutil.copyfile(path,target)
    with pymupdf.open(path) as doc:
        if not doc.is_pdf or doc.needs_pass or not 0 < len(doc) <= settings.max_pdf_pages:
            raise ValueError("Use an unlocked PDF within the configured page limit")
        product = Product(id=product_id,name=name.strip(),model=model.strip(),pdf_sha256=digest,n_pages=len(doc))
        blocks = extract_blocks(doc,settings.heading_size_ratio)
        if not blocks:
            raise ValueError("This PDF has no selectable text. Stage 1 requires a text-layer manual.")
        chunks = chunk_blocks(blocks,product,settings.chunk_tokens)
        pages = []
        figures = []
        for i,page in enumerate(doc):
            pages.append(render_page(page,product.id,folder/"pages",settings))
            section = next((b.section_path for b in blocks if b.page_no==i+1),"Manual")
            chunks.extend(extract_tables(page,product,section,len(chunks)))
            figures.extend(extract_figures(page,product,folder/"figures",settings.figure_min_size))
            if settings.enable_page_cards:
                if llm is None:
                    raise ValueError("Page cards require an LLM client and API key")
                card=page_card(product,pages[-1],llm,settings)
                pages[-1].page_card_json=card.model_dump_json()
                chunks.append(make_chunk(product,i+1,section,card.model_dump_json(),len(chunks),"page_card",bbox_source="none"))
                for warning in card.warning_texts:
                    if warning and warning in pages[-1].text and not any(c.is_warning and warning in c.text for c in chunks if c.page_no==i+1):
                        chunks.append(make_chunk(product,i+1,section,warning,len(chunks),"warning",bbox_source="none"))
            if progress:
                progress((i+1)/len(doc),f"Processed page {i+1} of {len(doc)}")
        if not any(p.text.strip() for p in pages):
            raise ValueError("This PDF has no text layer. Stage 1 requires a selectable-text manual; scanned-page OCR needs a separately validated transcription pipeline.")
        errors=extract_error_codes(product,pages,chunks)
        Store(settings.db_path).save_bundle(product,pages,chunks,figures,errors)
    logger.info("Ingested %s %s: %s pages, %s chunks",name,model,len(pages),len(chunks))
    return product
