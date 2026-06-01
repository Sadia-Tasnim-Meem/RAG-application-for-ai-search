from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
_cache: dict[str, dict] = {}


def load_prompt(name: str = "default") -> ChatPromptTemplate:
    if not _cache:
        raw = yaml.safe_load(_PROMPTS_PATH.read_text())
        _cache.update(raw)
    return ChatPromptTemplate.from_template(_cache[name]["template"])
