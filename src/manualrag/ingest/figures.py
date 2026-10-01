"""Detect raster and vector figures using PDF geometry only."""
from pathlib import Path
import pymupdf
from manualrag.models import Figure, Product

def extract_figures(page: pymupdf.Page, product: Product, folder: Path, min_size: float = 20) -> list[Figure]:
    """Union overlapping image/drawing rectangles, excluding detected table regions."""
    tables=[pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    boxes=[pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    boxes += list(page.cluster_drawings())
    merged: list[pymupdf.Rect]=[]
    for box in boxes:
        box &= page.rect
        if box.width<min_size or box.height<min_size:
            continue
        if any((box&t).get_area()/max(box.get_area(),1)>0.8 for t in tables):
            continue
        overlaps=[r for r in merged if r.intersects(box)]
        for other in overlaps:
            box |= other
            merged.remove(other)
        merged.append(box)
    folder.mkdir(parents=True,exist_ok=True)
    result=[]
    for index,box in enumerate(merged):
        identifier=f"{product.id}-p{page.number+1}-f{index+1}"
        crop=folder/f"{identifier}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(2,2),clip=box,alpha=False).save(crop)
        caption=page.get_text("text",clip=box).strip()
        result.append(Figure(figure_id=identifier,product_id=product.id,page_no=page.number+1,
                             bbox=tuple(box),crop_path=str(crop.resolve()),caption=caption))
    return result
