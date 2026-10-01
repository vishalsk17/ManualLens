"""Validated settings; all paths are relative to the configuration file."""
from pathlib import Path
from typing import Literal
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
import yaml


class Settings(BaseModel):
    """Runtime settings shared by ingestion, retrieval, generation, and evaluation."""

    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Path("data")
    answer_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-4-5-20251001"
    vision_model: str = "claude-sonnet-5-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_revision: str | None = None
    model_prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank: bool = False
    vector_backend: Literal["numpy", "faiss"] = "numpy"
    enable_page_cards: bool = False
    enable_stage2: bool = False
    prompt_version: str = "v1"
    render_zoom: float = Field(2.0, gt=0)
    thumbnail_width: int = Field(220, gt=0)
    chunk_tokens: int = Field(350, ge=50)
    bm25_k: int = Field(30, gt=0)
    dense_k: int = Field(30, gt=0)
    rrf_k: int = Field(60, gt=0)
    candidate_k: int = Field(20, gt=0)
    final_k: int = Field(6, gt=0)
    per_page_k: int = Field(3, gt=0)
    neighbor_chunks: int = Field(1, ge=0)
    max_evidence_chunks: int = Field(40, gt=0)
    gate_threshold: float = Field(0.08, ge=0, le=1)
    gate_calibrated: bool = False
    history_turns: int = Field(6, gt=0)
    visual_pages: int = Field(2, ge=1, le=2)
    max_output_tokens: int = Field(3000, gt=0)
    max_upload_mb: int = Field(50, gt=0)
    max_pdf_pages: int = Field(300, gt=0)
    heading_size_ratio: float = Field(1.15, gt=1)
    figure_min_size: float = Field(20, gt=0)
    request_timeout_seconds: int = Field(90, gt=0)
    input_cost_per_million: float | None = Field(None, ge=0)
    output_cost_per_million: float | None = Field(None, ge=0)
    stage1_recall5_min: float = 0.8
    stage1_abstention_f1_min: float = 0.8
    stage1_citation_precision_min: float = 1.0
    stage1_warning_recall_min: float = 1.0
    stage1_faithfulness_min: float = 0.95
    seed: int = 42

    @property
    def db_path(self) -> Path:
        """Return the persistent database path."""
        return self.data_dir / "manualrag.sqlite"

    @property
    def cache_dir(self) -> Path:
        """Return the cache root."""
        return self.data_dir / "cache"


def load_settings(path: Path = Path("config.yaml")) -> Settings:
    """Load YAML and optional model overrides without exposing API credentials."""
    path = path.resolve()
    load_dotenv(path.parent / ".env", override=False)
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name in ("answer_model", "fast_model", "vision_model"):
        if os.getenv(name.upper()):
            values[name] = os.environ[name.upper()]
    values["data_dir"] = (path.parent / values.get("data_dir", "data")).resolve()
    settings = Settings(**values)
    if settings.enable_stage2:
        raise ValueError("Stage 2 modules are pending Stage 1 real-manual evaluation and audit; leave enable_stage2 false.")
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    return settings
