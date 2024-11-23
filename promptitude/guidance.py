# original guidance __version__ = "0.0.64"

import types
import sys
import os
import requests
from . import library as commands
from ._program import Program
from . import llms

from ._utils import load, chain
from . import selectors
import nest_asyncio
import asyncio

# the user needs to set an LLM before they can use guidance
llm = None


# This makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, template=None, llm=None, initial_state=None, cache_seed=0, logprobs=None, silent=None, async_mode=False, stream=None, caching=None, await_missing=False, logging=False, **kwargs):
        return Program(text=template, llm=llm, initial_state=initial_state, cache_seed=cache_seed, logprobs=logprobs, silent=silent, async_mode=async_mode, stream=stream, caching=caching, await_missing=await_missing, logging=logging, **kwargs)


sys.modules[__name__].__class__ = Guidance
