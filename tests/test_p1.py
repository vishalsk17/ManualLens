from pathlib import Path
import pymupdf
from PIL import Image
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.index.store import Store
from manualrag.chunking import split_procedure
from manualrag.config import Settings

def test_ingest_geometry_sections_and_tables(manual: Path, settings: Settings) -> None:
    product=ingest_pdf(manual,"Lab printer","ML-100",settings)
    store=Store(settings.db_path)
    assert product.n_pages==6
    pages=store.pages(product.id)
    assert [p.page_no for p in pages]==[1,2,3,4,5,6]
    with Image.open(pages[0].png_path) as image:
        assert image.size==(1190,1684)
    chunks=store.chunks(product.id)
    assert any(c.type=="table" and "E04" in c.text for c in chunks)
    assert any(c.is_warning and c.page_no==4 and "Paper jams" in c.section_path for c in chunks)
    procedure=next(c for c in chunks if c.text.startswith("1. Remove"))
    assert "4. Press Power" in procedure.text
    for c in chunks:
        assert c.bbox_source=="pdf"
        x0,y0,x1,y1=c.bbox
        assert 0<=x0<x1<=595 and 0<=y0<y1<=842
    ingest_pdf(manual,"Lab printer","ML-100",settings)
    assert len(store.chunks(product.id))==len(chunks)

def test_procedure_split_preserves_steps() -> None:
    text="\n".join(f"{i}. "+("word "*30) for i in range(1,7))
    parts=split_procedure(text,85)
    assert len(parts)>1
    assert all(p[0].isdigit() for p in parts)
    assert sum(p.count("word") for p in parts)==180

def test_rotated_page_render_matches_pdf_coordinates(tmp_path: Path, settings: Settings) -> None:
    path=tmp_path/"rotated.pdf"
    doc=pymupdf.open()
    p=doc.new_page(width=200,height=300)
    p.insert_text((20,50),"Test button")
    p.set_rotation(90)
    doc.save(path)
    doc.close()
    product=ingest_pdf(path,"Test","T1",settings)
    page=Store(settings.db_path).page(product.id,1)
    assert (page.width,page.height)==(200,300)
    with Image.open(page.png_path) as im:assert im.size==(400,600)
