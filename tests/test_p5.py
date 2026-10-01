from pathlib import Path
from streamlit.testing.v1 import AppTest
from manualrag.config import Settings
from manualrag.ingest.pipeline import ingest_pdf
APP=Path(__file__).resolve().parents[1]/"app"/"streamlit_app.py"

def test_empty_ui_and_manual_page_navigation(tmp_path: Path,manual: Path,monkeypatch) -> None:
    config=tmp_path/"config.yaml"
    config.write_text("data_dir: data\n",encoding="utf-8")
    monkeypatch.setenv("MANUALLENS_CONFIG",str(config))
    monkeypatch.delenv("ANTHROPIC_API_KEY",raising=False)
    app=AppTest.from_file(str(APP),default_timeout=45).run()
    assert not app.exception
    assert any("Upload a PDF" in x.value for x in app.info)
    settings=Settings(data_dir=tmp_path/"data")
    ingest_pdf(manual,"Lab printer","ML-100",settings)
    app=AppTest.from_file(str(APP),default_timeout=45).run()
    assert not app.exception
    page_select=next(s for s in app.selectbox if s.label=="PDF page")
    page_select.select(4).run()
    assert not app.exception
    assert app.session_state["viewer_page"]==4
    assert any("Paper jams" in x.value for x in app.text)
