"""Dataset schemas and leakage-resistant fixed split validation."""
from pathlib import Path
from hashlib import sha256
from typing import Literal
import json
from pydantic import Field,model_validator
from manualrag.models import Schema

class HistoryEntry(Schema):
    """A supplied prior conversation turn, not additional product evidence."""
    question: str
    answer: str
    pages: list[int] = Field(default_factory=list)

class Question(Schema):
    """One annotated manual question with a fixed DEV/TEST assignment."""
    id: str
    question: str
    category: str
    answerable: bool
    gold_pages: list[int]
    gold_points: list[str]
    needs_visual: bool=False
    history: list[HistoryEntry]=Field(default_factory=list)
    split: Literal["dev","test"]

    @model_validator(mode="after")
    def check_gold(self) -> "Question":
        """Reject contradictory answerability and page annotations."""
        if self.answerable and (not self.gold_pages or not self.gold_points):
            raise ValueError("Answerable questions need gold pages and points")
        if any(p<1 for p in self.gold_pages):raise ValueError("Pages are one-based")
        if not self.answerable and self.gold_pages:raise ValueError("Unanswerable questions cannot have gold answer pages")
        return self

def load_dataset(path: Path) -> tuple[list[Question],str,dict]:
    """Validate unique IDs, splits, and a dataset provenance manifest."""
    records=[Question.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({q.id for q in records})!=len(records):raise ValueError("Duplicate question IDs")
    normalized=[q.question.strip().casefold() for q in records]
    if len(set(normalized))!=len(normalized):raise ValueError("Duplicate questions risk split leakage")
    for split in ("dev","test"):
        group=[q for q in records if q.split==split]
        if not group or len({q.answerable for q in group})<2:
            raise ValueError(f"{split} needs answerable and unanswerable questions")
    manifest_path=path.with_suffix(".manifest.json")
    if not manifest_path.exists():raise ValueError("Dataset provenance manifest is required")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_type") not in {"synthetic","real"}:
        raise ValueError("Dataset source_type must be synthetic or real")
    return records,sha256(path.read_bytes()).hexdigest(),manifest
