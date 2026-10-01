from pathlib import Path
import pymupdf
from manualrag.config import Settings
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.index.store import Store
from manualrag.ingest.error_codes import normalize_code, validate_error_row
from manualrag.ingest.figures import extract_figures
from manualrag.ingest.vlm_cards import page_card
from manualrag.llm.client import Completion
from manualrag.models import ErrorCode
import pytest

class CardClient:
    def __init__(self) -> None:self.calls=0
    def complete(self, **kwargs: object) -> Completion:
        self.calls+=1
        return Completion('{"summary":"Control panel","figures":[],"labels":{"Power":"left"},"warning_texts":[],"uncertainty_notes":[]}')

@pytest.mark.parametrize("value",["E04","E-04","e 04"])
def test_code_normalization(value: str) -> None:
    assert normalize_code(value)=="E04"

def test_unverifiable_error_rejected() -> None:
    row=ErrorCode(product_id="x",code_norm="E99",code_raw="E99",meaning="Invented",page_no=1,chunk_id="x")
    assert not validate_error_row(row,"E04 Paper jam","x")

def test_figures_codes_and_cache(manual: Path,settings: Settings) -> None:
    product=ingest_pdf(manual,"Lab printer","ML-100",settings)
    store=Store(settings.db_path)
    assert store.errors(product.id,"E04")[0].meaning=="Paper jam"
    assert store.errors(product.id,"F12")[0].code_raw=="F-12"
    assert len(store.figures(product.id,2))>=1
    client=CardClient()
    page=store.page(product.id,2)
    first=page_card(product,page,client,settings)
    second=page_card(product,page,client,settings)
    assert first==second and client.calls==1
    changed=settings.model_copy(update={"prompt_version":"v2"})
    page_card(product,page,client,changed)
    assert client.calls==2
    assert not store.errors("another-model","E04")
