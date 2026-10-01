"""Render PDFs with PyMuPDF; no external executables required."""
from pathlib import Path
import pymupdf
from PIL import Image
from manualrag.config import Settings
from manualrag.models import Page

def render_page(page: pymupdf.Page, product_id: str, folder: Path, settings: Settings) -> Page:
    """Render unrotated coordinates so PDF bboxes map directly to pixels."""
    page.set_rotation(0)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{page.number + 1:04d}.png"
    thumb = folder / f"{page.number + 1:04d}.thumb.png"
    page.get_pixmap(matrix=pymupdf.Matrix(settings.render_zoom, settings.render_zoom), alpha=False).save(png)
    with Image.open(png) as image:
        image.thumbnail((settings.thumbnail_width, settings.thumbnail_width * 2))
        image.save(thumb)
    return Page(product_id=product_id,page_no=page.number+1,png_path=str(png.resolve()),
                thumb_path=str(thumb.resolve()),width=page.rect.width,height=page.rect.height,
                text=page.get_text("text"))
