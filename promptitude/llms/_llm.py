from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Type
from types import TracebackType

import asyncio
import re
import json
import inspect
import threading
from abc import ABCMeta, abstractmethod

from promptitude import guidance

from .caches import DiskCache


class LLMMeta(ABCMeta):
    """ Metaclass for LLM classes to manage a shared DiskCache instance.

    Ensures that all instances of an LLM subclass share the same cache,
    facilitating caching across different instances.
    """

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs) -> None:
        super().__init__(name, bases, namespace)
        cls._cache = None
        cls._lock = threading.Lock()

    @property
    def cache(cls) -> DiskCache:
        with cls._lock:
            if cls._cache is None:
                cls._cache = DiskCache(cls.llm_name)
        return cls._cache

    @cache.setter
    def cache(cls, value: DiskCache) -> None:
        with cls._lock:
            cls._cache = value


class LLM(metaclass=LLMMeta):
    """ Base class for Language Model interfaces.

    Provides a common interface and shared functionality for different LLM implementations.
    """
    cache_version: int = 1  # Version of the cache to handle cache invalidation when the class implementation changes.
    default_system_prompt: str = "You are a helpful assistant."
    llm_name: str = "unknown"

    # Serialization
    excluded_args: List[str] = []
    class_attribute_map: Dict[str, str] = {}

    def __init__(self) -> None:
        self.chat_mode = False  # by default models are not in role-based chat mode
        self.model_name = "unknown"

        # these should all start with the @ symbol and are variables programs can use when running with this LLM
        self.tool_def = guidance("""
# Tools

{{#if len(functions) > 0~}}
## functions

namespace functions {

{{#each functions item_name="function"~}}
// {{function.description}}
type {{function.name}} = (_: {
{{~#each function.parameters.properties}}
{{#if contains(this, "description")}}// {{this.description}}
{{/if~}}
{{@key}}{{#unless contains(function.parameters.required, @key)}}?{{/unless}}: {{#if contains(this, "enum")}}{{#each this.enum}}"{{this}}"{{#unless @last}} | {{/unless}}{{/each}}{{else}}{{this.type}}{{/if}}{{#unless @last}},{{/unless}}
{{~/each}}
}) => any;

{{/each~}}
} // namespace functions
{{~/if~}}""", functions=[])
        self.function_call_stop_regex = r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```"

    def __call__(self, *args, **kwargs) -> Any:
        """Generates a response from the LLM. Subclasses must implement this method."""
        raise NotImplementedError("LLM subclasses must implement the __call__ method.")

    def __getitem__(self, key: str) -> Any:
        """Gets an attribute from the LLM."""
        return getattr(self, key)

    def session(self, asynchronous: bool = False) -> Union[LLMSession, SyncSession]:
        """Creates a session for the LLM.

        This implementation is meant to be overridden by subclasses.
        """

        if asynchronous:
            return LLMSession(self)
        else:
            return SyncSession(LLMSession(self))

    @staticmethod
    def extract_function_call(text: str) -> Optional[CallableAnswer]:
        """Extracts a callable function from the LLM's output, if any."""
        m = re.match(r"\n?\n?```typescript\nfunctions.([^\(]+)\((.*?)\)```", text, re.DOTALL)
        if m:
            return CallableAnswer(m.group(1), m.group(2))

    @abstractmethod
    def encode(self, string: str, **kwargs) -> List[int]:
        """Abstract method to encode a string into tokens. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int], **kwargs) -> str:
        """Abstract method to decode tokens into a string. Must be implemented by subclasses."""
        pass

    def id_to_token(self, id: int) -> str:
        return self.decode([id])

    def token_to_id(self, token: str) -> int:
        return self.encode(token)[0]

    # allow for caches to be get and set on the object as well as the class
    @property
    def cache(self) -> DiskCache:
        if self._cache is not None:
            return self._cache
        else:
            return self.__class__.cache

    @cache.setter
    def cache(self, value: DiskCache) -> None:
        self._cache = value

    def serialize(self) -> Dict[str, Any]:
        """Serializes the LLM instance for caching or storage purposes"""
        excluded_args = set(self.excluded_args)
        class_attribute_map = self.class_attribute_map
        init_params = inspect.signature(self.__init__).parameters
        out = {
            'module_name': self.__module__,
            'class_name': self.__class__.__name__,
            'init_args': {
                arg: getattr(self, class_attribute_map.get(arg, arg))
                for arg in init_params
                if arg not in excluded_args and hasattr(self, class_attribute_map.get(arg, arg))
            }
        }
        return out


class LLMSession:
    """Asynchronous session class for interacting with the LLM.

    Manages stateful interactions with the LLM, such as tracking call counts
    for non-zero temperature requests, and caching considerations.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._call_counts: Dict = {}  # tracks the number of repeated identical calls to the LLM with non-zero temperature

    def __enter__(self) -> LLMSession:
        return self

    async def __call__(self, *args, **kwargs) -> Any:
        return self.llm(*args, **kwargs)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        pass

    def _gen_key(self, args_dict: Dict[str, Any]) -> str:
        """Generates a unique key for caching based on the arguments."""
        del args_dict["self"]  # skip the "self" arg
        return "_---_".join([str(v) for v in (
                [args_dict[k] for k in args_dict] + [self.llm.model_name, self.llm.__class__.__name__,
                                                     self.llm.cache_version])])

    def _cache_params(self, args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares cache parameters, including call counts for non-deterministic outputs."""
        key = self._gen_key(args_dict)
        # if we have non-zero temperature we include the call count in the cache key
        if args_dict.get("temperature", 0) > 0:
            args_dict["call_count"] = self._call_counts.get(key, 0)

            # increment the call count
            self._call_counts[key] = args_dict["call_count"] + 1
        args_dict["model_name"] = self.llm.model_name
        args_dict["cache_version"] = self.llm.cache_version
        args_dict["class_name"] = self.llm.__class__.__name__

        return args_dict


class SyncSession:
    """Synchronous wrapper for LLMSession."""

    def __init__(self, session: LLMSession) -> None:
        self._session = session

    def __enter__(self) -> SyncSession:
        self._session.__enter__()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        return self._session.__exit__(exc_type, exc_value, traceback)

    def __call__(self, *args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._session.__call__(*args, **kwargs))
        else:
            # If the event loop is already running, schedule the coroutine and wait for it
            future = asyncio.ensure_future(self._session.__call__(*args, **kwargs))
            return loop.run_until_complete(future)


class CallableAnswer:
    """Represents a callable function extracted from the LLM's output."""

    def __init__(self, name: str, args_string: str, function: Optional[Any] = None) -> None:
        self.__name__: str = name
        self.args_string: str = args_string
        self._function: Optional[Any] = function

    def __call__(self, *args, **kwargs) -> Any:
        if self._function is None:
            raise NotImplementedError(f"Answer {self.__name__} has no function defined")
        return self._function(*args, **self.__kwdefaults__, **kwargs)

    @property
    def __kwdefaults__(self) -> Dict[str, Any]:
        """Parses and returns the default keyword arguments from the arguments string."""
        # We build this lazily in case the user wants to handle validation errors themselves.
        return json.loads(self.args_string)

    def __repr__(self) -> str:
        return f"CallableAnswer(__name__={self.__name__}, __kwdefaults__={self.__kwdefaults__})"
