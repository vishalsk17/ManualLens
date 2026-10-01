"""PDF-coordinate highlights; estimated VLM boxes never become precise overlays."""
from pathlib import Path
from hashlib import sha256
from PIL import Image,ImageDraw
from manualrag.index.store import Store
from manualrag.models import Answer,Hit,Page,VisualPayload,BBox

def highlight_page(page: Page,hits: list[Hit],output: Path) -> tuple[Path,list[BBox]]:
    """Scale exact PDF rectangles to the actual rendered page pixel size."""
    with Image.open(page.png_path) as source:
        image=source.convert("RGBA")
    layer=Image.new("RGBA",image.size,(0,0,0,0))
    draw=ImageDraw.Draw(layer)
    sx,sy=image.width/page.width,image.height/page.height
    rectangles=[]
    for hit in hits:
        c=hit.chunk
        if c.page_no!=page.page_no or c.product_id!=page.product_id or c.bbox_source!="pdf" or c.bbox is None:
            continue
        x0,y0,x1,y1=c.bbox
        box=(max(0,x0*sx),max(0,y0*sy),min(image.width,x1*sx),min(image.height,y1*sy))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        rectangles.append(box)
        color=(217,119,6,55) if c.is_warning else (37,99,235,45)
        draw.rectangle(box,fill=color,outline=(*color[:3],220),width=3)
    output.parent.mkdir(parents=True,exist_ok=True)
    Image.alpha_composite(image,layer).convert("RGB").save(output)
    return output,rectangles

def visual_payloads(answer: Answer,hits: list[Hit],store: Store,product_id: str,folder: Path) -> list[VisualPayload]:
    """Build page images only from verified citations; abstentions show related pages unhighlighted."""
    cited={i for s in answer.steps for i in s.chunk_ids}|{w.chunk_id for w in answer.warnings}
    selected=[h for h in hits if h.chunk.chunk_id in cited]
    pages=list(dict.fromkeys(h.chunk.page_no for h in selected))
    if not pages:pages=list(dict.fromkeys(h.chunk.page_no for h in hits))[:2]
    payload=[]
    for number in pages:
        page=store.page(product_id,number)
        page_hits=[h for h in selected if h.chunk.page_no==number]
        ids=[h.chunk.chunk_id for h in page_hits]
        key=sha256((product_id+str(number)+"|".join(sorted(ids))).encode("utf-8")).hexdigest()[:24]
        path,rectangles=highlight_page(page,page_hits,folder/f"{key}.png")
        figure_ids={h.chunk.figure_id for h in page_hits if h.chunk.figure_id}
        crops=[f.crop_path for f in store.figures(product_id,number) if f.figure_id in figure_ids]
        payload.append(VisualPayload(page_no=number,png_path=page.png_path,overlay_path=str(path.resolve()),
                                     chunk_ids=ids,rectangles=rectangles,figure_paths=crops))
    return payload
