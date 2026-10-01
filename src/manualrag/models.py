"""Strict schemas at all model and persistence boundaries."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

BBox = tuple[float, float, float, float]
ChunkType = Literal["text", "table", "figure", "page_card", "warning", "error_row"]
Status = Literal["answered", "partial", "not_in_manual", "needs_clarification"]


class Schema(BaseModel):
    """Reject unexpected keys in model responses."""
    model_config = ConfigDict(extra="forbid")


class Product(Schema):
    id: str
    name: str
    model: str
    pdf_sha256: str
    n_pages: int = Field(gt=0)


class Page(Schema):
    product_id: str
    page_no: int = Field(ge=1)
    png_path: str
    thumb_path: str
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    page_card_json: str | None = None
    text: str = ""


class Block(Schema):
    page_no: int = Field(ge=1)
    text: str
    bbox: BBox
    font_size: float
    bold: bool = False
    section_path: str


class Chunk(Schema):
    chunk_id: str
    product_id: str
    page_no: int = Field(ge=1)
    type: ChunkType = "text"
    section_path: str
    text: str
    embed_text: str
    bbox: BBox | None = None
    bbox_source: Literal["pdf", "vlm", "none"] = "none"
    is_warning: bool = False
    figure_id: str | None = None
    ordinal: int = 0

    @model_validator(mode="after")
    def validate_bbox(self) -> "Chunk":
        """Reject inverted or unlabelled geometry."""
        if self.bbox:
            x0, y0, x1, y1 = self.bbox
            if x1 <= x0 or y1 <= y0 or self.bbox_source == "none":
                raise ValueError("Invalid bounding box or missing provenance")
        return self


class Figure(Schema):
    figure_id: str
    product_id: str
    page_no: int
    bbox: BBox
    crop_path: str
    caption: str = ""
    callouts: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class ErrorCode(Schema):
    product_id: str
    code_norm: str
    code_raw: str
    meaning: str
    cause: str = ""
    remedy: str = ""
    page_no: int
    chunk_id: str


class FigureCard(Schema):
    caption: str
    description: str
    callouts: dict[str, str] = Field(default_factory=dict)
    location: str = "unclear"


class PageCard(Schema):
    summary: str
    figures: list[FigureCard] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    warning_texts: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


class QueryPlan(Schema):
    standalone_query: str
    intent: Literal["locate_part", "how_to", "error_code", "troubleshoot", "spec_lookup", "out_of_scope"]
    error_code: str | None = None
    component: str | None = None
    needs_visual: bool = False
    other_model_mentioned: str | None = None


class Step(Schema):
    n: int = Field(ge=1)
    text: str
    chunk_ids: list[str] = Field(min_length=1)


class Warning(Schema):
    text: str
    chunk_id: str


class VisualRequest(Schema):
    page_no: int = Field(ge=1)
    kind: Literal["page", "figure"] = "page"
    reason: str


class Answer(Schema):
    status: Status
    summary: str
    steps: list[Step] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    visuals: list[VisualRequest] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    missing_info: list[str] = Field(default_factory=list)


class Hit(Schema):
    chunk: Chunk
    score: float
    lexical_score: float = 0
    dense_score: float = 0
    role: Literal["retrieved", "exact", "warning", "neighbor"] = "retrieved"


class GateDecision(Schema):
    allowed: bool
    status: Status
    score: float
    reason: str


class VisualPayload(Schema):
    page_no: int
    png_path: str
    overlay_path: str
    chunk_ids: list[str]
    rectangles: list[BBox]
    figure_paths: list[str] = Field(default_factory=list)


class Turn(Schema):
    question: str
    answer: Answer
    cited_pages: list[int] = Field(default_factory=list)


class ActiveContext(Schema):
    pages: list[int] = Field(default_factory=list)
    component: str | None = None
    error_code: str | None = None


class AskResult(Schema):
    conversation_id: str
    answer: Answer
    visuals: list[VisualPayload]
    trace: dict
