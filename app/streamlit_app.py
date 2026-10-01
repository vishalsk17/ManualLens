"""ManualLens Streamlit UI; product logic lives in manualrag."""
from pathlib import Path
import os
import sys
import hashlib
import logging
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from manualrag.config import load_settings
from manualrag.index.store import Store
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.llm.client import AnthropicClient
from manualrag.service import ManualAssistant
from manualrag.models import AskResult

logger=logging.getLogger(__name__)
st.set_page_config(page_title="ManualLens",page_icon=":material/menu_book:",layout="wide")
st.markdown("""
<style>
:root {--ml-ink:#16324f;--ml-link:#2458aa;--ml-border:#d6e0ea;}
[data-testid="stAppViewContainer"] {background:#f8fafc;}
h1,h2,h3 {color:var(--ml-ink);letter-spacing:-.025em;}
[data-testid="stSidebar"] {background:#eef3f8;}
.stButton>button {min-height:44px;border-color:var(--ml-border);}
.stButton>button:focus-visible {outline:3px solid var(--ml-link);outline-offset:2px;}
[data-testid="stChatMessage"] {background:white;border:1px solid var(--ml-border);border-radius:12px;}
[data-testid="stCaptionContainer"] {color:#475569;}
@media(max-width:760px){[data-testid="stHorizontalBlock"]{flex-wrap:wrap;}}
</style>
""",unsafe_allow_html=True)
settings=load_settings(Path(os.getenv("MANUALLENS_CONFIG",str(ROOT/"config.yaml"))))
store=Store(settings.db_path)

def select_page(page_no: int,result_index: int) -> None:
    """Navigate a citation to its exact page and answer highlights."""
    st.session_state["viewer_page"]=page_no
    st.session_state["viewer_result"]=result_index

def render_answer(result: AskResult,index: int) -> None:
    """Show verified facts, warnings, and citation buttons."""
    answer=result.answer
    status=answer.status.replace("_"," ").title()
    if answer.status=="answered":st.success(status)
    elif answer.status=="partial":st.warning(status)
    else:st.info(status)
    st.write(answer.summary)
    chunk_pages={h["chunk"]["chunk_id"]:h["chunk"]["page_no"] for h in result.trace["retrieved"]}
    for step in answer.steps:
        st.write(f"{step.n}. {step.text}")
        pages=list(dict.fromkeys(chunk_pages[i] for i in step.chunk_ids))
        cols=st.columns(max(1,min(len(pages),4)))
        for j,page in enumerate(pages):
            cols[j%len(cols)].button(f"p.{page}",key=f"cite-{index}-{step.n}-{page}",
                                     on_click=select_page,args=(page,index),help=f"Open physical PDF page {page}")
    for j,warning in enumerate(answer.warnings):
        st.warning(warning.text,icon=":material/warning:")
        page=chunk_pages[warning.chunk_id]
        st.button(f"Warning · p.{page}",key=f"warn-{index}-{j}",on_click=select_page,args=(page,index))
    for missing in answer.missing_info:st.caption(missing)
    if answer.status in {"not_in_manual","needs_clarification"} and result.visuals:
        st.caption("Possibly related pages — these do not establish an answer.")
        for v in result.visuals:
            st.button(f"Possibly related · p.{v.page_no}",key=f"related-{index}-{v.page_no}",on_click=select_page,args=(v.page_no,index))
    with st.expander("Debug trace"):st.json(result.trace,expanded=False)

with st.sidebar:
    st.title("ManualLens")
    st.caption("Your manual. Clear answers.")
    with st.expander("Add a product manual",expanded=not store.products()):
        uploaded=st.file_uploader("PDF manual",type=["pdf"],help="One product model per manual.")
        name=st.text_input("Product name",placeholder="Example: Office printer")
        model=st.text_input("Model",placeholder="Example: ML-100")
        if st.button("Process manual",type="primary",disabled=uploaded is None or not name.strip() or not model.strip()):
            progress=st.progress(0,text="Preparing manual")
            try:
                data=uploaded.getvalue()
                if len(data)>settings.max_upload_mb*1024*1024:raise ValueError("PDF exceeds configured upload size.")
                folder=settings.data_dir/"manuals"
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/(hashlib.sha256(data).hexdigest()+".pdf")
                path.write_bytes(data)
                llm=AnthropicClient(settings) if settings.enable_page_cards else None
                p=ingest_pdf(path,name,model,settings,llm=llm,progress=lambda n,text:progress.progress(n,text=text))
                for key in ("conversation_id","results","questions","viewer_page","viewer_result","assistant"):
                    st.session_state.pop(key,None)
                st.session_state["active_product"]=p.id
                st.success(f"Ready: {p.n_pages} pages")
                st.rerun()
            except Exception as exc:
                logger.exception("Ingestion failed")
                st.error(f"Could not process this manual: {exc}")
    products=store.products()
    if not products:
        st.info("Upload a PDF to begin.")
        st.stop()
    labels={p.id:f"{p.name} · {p.model}" for p in products}
    ids=list(labels)
    previous=st.session_state.get("active_product")
    product_id=st.selectbox("Active manual",ids,format_func=lambda x:labels[x],index=ids.index(previous) if previous in ids else 0)
    product=store.product(product_id)
    st.caption(f"{product.n_pages} pages · PDF page numbers start at 1")
    if st.button("New conversation"):
        st.session_state.pop("conversation_id",None)
        st.session_state["results"]=[]
        st.session_state["questions"]=[]
        st.session_state["viewer_page"]=1
    st.divider()
    st.caption("Answers use this manual only. Blue highlights mark evidence; amber highlights mark warnings.")
if st.session_state.get("active_product")!=product_id:
    for key in ("conversation_id","results","questions","viewer_page","viewer_result","assistant"):
        st.session_state.pop(key,None)
    st.session_state["active_product"]=product_id
st.session_state.setdefault("results",[])
st.session_state.setdefault("questions",[])
st.session_state.setdefault("viewer_page",1)
st.session_state.setdefault("viewer_result",0)
st.title("Find the answer. See the evidence.")
st.caption(f"{product.name} / {product.model}")
left,right=st.columns([1.05,1],gap="large")
with left:
    st.subheader("Ask your manual")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.info("Page browsing is ready. Add ANTHROPIC_API_KEY to the project's .env file and restart to enable answers.")
    if not st.session_state["results"]:
        st.caption("Try asking about setup, a control-panel light, an error code, or a paper jam.")
    for index,(question,result) in enumerate(zip(st.session_state["questions"],st.session_state["results"])):
        with st.chat_message("user"):st.write(question)
        with st.chat_message("assistant"):render_answer(result,index)
    question=st.chat_input("Ask a question about this model",disabled=not bool(os.getenv("ANTHROPIC_API_KEY")))
    if question:
        with st.spinner("Finding supporting pages and checking citations…"):
            try:
                if "assistant" not in st.session_state:
                    st.session_state["assistant"]=ManualAssistant(settings,AnthropicClient(settings))
                result=st.session_state["assistant"].ask(product_id,question,st.session_state.get("conversation_id"))
                st.session_state["conversation_id"]=result.conversation_id
                st.session_state["results"].append(result)
                st.session_state["questions"].append(question)
                if result.visuals:st.session_state["viewer_page"]=result.visuals[0].page_no
                st.session_state["viewer_result"]=len(st.session_state["results"])-1
                st.rerun()
            except Exception as exc:
                logger.exception("Question failed")
                st.error(f"No answer was generated: {exc}")
with right:
    st.subheader("Manual evidence")
    st.selectbox("PDF page",list(range(1,product.n_pages+1)),key="viewer_page",format_func=lambda n:f"Page {n}")
    number=st.session_state["viewer_page"]
    page=store.page(product_id,number)
    shown=page.png_path
    selected=None
    results=st.session_state["results"]
    if results:
        index=min(st.session_state["viewer_result"],len(results)-1)
        selected=next((v for v in results[index].visuals if v.page_no==number),None)
        if selected:shown=selected.overlay_path
    st.image(shown,caption=f"{product.model} · physical PDF page {number}",width="stretch")
    if selected and selected.rectangles:st.caption("Highlighted regions use exact PDF text geometry.")
    with st.expander("Accessible page text"):st.text(page.text or "This page has no selectable text layer.")
    figures=store.figures(product_id,number)
    if figures:
        with st.expander(f"Detected graphics ({len(figures)})"):
            for f in figures:st.image(f.crop_path,caption=f.caption or "Graphic extracted from this page",width="stretch")
    pdf=settings.cache_dir/"manuals"/product.pdf_sha256/"manual.pdf"
    if pdf.exists():st.download_button("Download original manual",pdf.read_bytes(),file_name=f"{product.model}-manual.pdf",mime="application/pdf")
