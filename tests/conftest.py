from pathlib import Path
import pytest
from manualrag.config import Settings
from manualrag.ingest.synthetic import create_synthetic_manual

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path/"data")

@pytest.fixture
def manual(tmp_path: Path) -> Path:
    return create_synthetic_manual(tmp_path/"fixture.pdf")
