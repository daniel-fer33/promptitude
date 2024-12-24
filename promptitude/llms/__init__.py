from ._openai import OpenAI
from ._anthropic import Anthropic
from ._transformers import Transformers
from ._mock import Mock
from ._llm import LLM, LLMSession, SyncSession
from . import transformers
from . import caches


def get_llm_from_model_name(model_name: str) -> LLM:
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return OpenAI(model_name, caching=False)
    elif model_name.startswith("claude-"):
        return Anthropic(model_name, caching=False)
    else:
        return Transformers(model_name, device='cpu', caching=False)
