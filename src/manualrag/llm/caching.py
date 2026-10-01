"""Deterministic content-addressed caches with atomic writes."""
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json

def cache_key(pdf_sha256: str, page: int, prompt_version: str, **extra: object) -> str:
    """Include provenance plus models/content so stale data cannot silently collide."""
    payload={"pdf_sha256":pdf_sha256,"page":page,"prompt_version":prompt_version,**extra}
    return sha256(json.dumps(payload,sort_keys=True,ensure_ascii=True).encode("utf-8")).hexdigest()

def atomic_text(path: Path, value: str) -> None:
    """Write UTF-8 using a same-directory temporary file and atomic replacement."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        temp.write_text(value,encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
