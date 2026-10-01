from pathlib import Path
import pytest
from pydantic import ValidationError
from manualrag.config import load_settings, Settings
from manualrag.models import Answer, Chunk


def test_config_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("data_dir: local\n", encoding="utf-8")
    settings = load_settings(tmp_path / "config.yaml")
    assert settings.data_dir == tmp_path / "local"
    assert settings.cache_dir.exists()


def test_models_fail_closed() -> None:
    with pytest.raises(ValidationError):
        Answer(status="sure", summary="x", confidence=2)
    with pytest.raises(ValidationError):
        Settings(final_k=0)
    with pytest.raises(ValidationError):
        Chunk(chunk_id="x", product_id="p", page_no=1, section_path="s", text="x", embed_text="x", bbox=(5, 5, 1, 1), bbox_source="pdf")
