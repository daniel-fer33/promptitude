import re

from ._openai import OpenAI
from ._anthropic import Anthropic
from ._transformers import Transformers
from ._mock import Mock
from ._llm import LLM, LLMSession, SyncSession
from . import transformers
from . import caches


def get_llm_from_model_name(model_name: str) -> LLM:
    if model_name.startswith('openai/'):
        # This indicates the use of OpenAI API schemas and SDK for other models
        model_name = model_name[len('openai/'):]  # Remove the "openai/" prefix
        return OpenAI(model_name, caching=False)
    if re.match(r"^(gpt-|o1|o3|chatgpt)", model_name):
        return OpenAI(model_name, caching=False)
    elif re.match(r"^claude-", model_name):
        return Anthropic(model_name, caching=False)
    else:
        return Transformers(model_name, device='cpu', caching=False)
