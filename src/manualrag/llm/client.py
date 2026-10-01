"""Small swappable LLM interface, with one schema-repair attempt."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar
import base64
import json
import os
from pydantic import BaseModel, ValidationError
from manualrag.config import Settings

T = TypeVar("T", bound=BaseModel)

@dataclass
class Completion:
    """Raw provider response and billable usage."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

class LLMClient(Protocol):
    """Provider contract; no provider-specific types leak into application code."""
    def complete(self, *, system: str, payload: dict, schema: dict, model: str,
                 images: list[Path] | None = None) -> Completion: ...

def untrusted_block(payload: dict) -> str:
    """Encode manual and conversation text as explicitly delimited untrusted JSON."""
    escaped = json.dumps(payload,ensure_ascii=True).replace("<",r"\u003c").replace(">",r"\u003e")
    return "BEGIN_UNTRUSTED_DATA\n"+escaped+"\nEND_UNTRUSTED_DATA"

class AnthropicClient:
    """Anthropic adapter; an absent key or provider error fails visibly."""
    def __init__(self, settings: Settings) -> None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY in .env to use live model answers.")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=key,timeout=settings.request_timeout_seconds,max_retries=1)
        self.settings = settings

    def complete(self, *, system: str, payload: dict, schema: dict, model: str,
                 images: list[Path] | None = None) -> Completion:
        """Request JSON and include page images only when explicitly supplied."""
        content: list[dict] = [{"type":"text","text":untrusted_block(payload)}]
        for path in images or []:
            content.append({"type":"image","source":{"type":"base64","media_type":"image/png",
                            "data":base64.b64encode(path.read_bytes()).decode("ascii")}})
        response = self.client.messages.create(
            model=model,max_tokens=self.settings.max_output_tokens,
            system=system+"\nReturn only JSON matching this schema:\n"+json.dumps(schema),
            messages=[{"role":"user","content":content}])
        text = "".join(b.text for b in response.content if b.type=="text")
        return Completion(text,response.usage.input_tokens,response.usage.output_tokens,model)

def structured_call(client: LLMClient, schema: type[T], *, system: str, payload: dict,
                    model: str, images: list[Path] | None = None,
                    usage: list[dict] | None = None) -> T:
    """Validate JSON with pydantic; retry once, then fail closed."""
    for attempt in range(2):
        result = client.complete(system=system,payload=payload,schema=schema.model_json_schema(),model=model,images=images)
        if usage is not None:
            usage.append({"model":model,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens})
        text = result.text.strip()
        if text.startswith("~~~"):
            text = text.strip("~").removeprefix("json").strip()
        # Accept a provider's Markdown JSON fence, but no prose outside it.
        if text.startswith(chr(96)*3):
            text=text[3:].removeprefix("json").strip()
            if text.endswith(chr(96)*3):text=text[:-3].strip()
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            if attempt:
                raise ValueError("Model response failed schema validation after one retry") from exc
            system += "\nYour previous response was invalid. Return a complete JSON object, no Markdown, with all required fields."
    raise AssertionError("Unreachable")
