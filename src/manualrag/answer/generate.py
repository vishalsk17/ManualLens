"""Answer generation with strict JSON validation."""
from pathlib import Path
from manualrag.answer.prompts import ANSWER_PROMPT
from manualrag.config import Settings
from manualrag.llm.client import LLMClient,structured_call
from manualrag.models import Answer,Hit,Product,QueryPlan

def generate_answer(plan: QueryPlan,hits: list[Hit],product: Product,client: LLMClient,
                    settings: Settings,images: list[Path],usage: list[dict]) -> Answer:
    """Generate only against retrieved evidence and explicitly attached page images."""
    return structured_call(client,Answer,system=ANSWER_PROMPT,
        payload={"product":product.model_dump(),"question":plan.standalone_query,
                 "evidence":[h.chunk.model_dump() for h in hits]},
        model=settings.answer_model,images=images,usage=usage)
