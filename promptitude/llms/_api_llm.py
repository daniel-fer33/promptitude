import copy
from typing import List, Dict, Optional, Any, Callable

from abc import abstractmethod
import time
import collections
import inspect
import asyncio

from ._llm import LLM, LLMSession


class APILLM(LLM):
    """Abstract base class for API-based LLMs."""
    _api_exclude_arguments: Optional[List[str]] = None  # Exclude arguments to pass to the API
    _api_rename_arguments: Optional[Dict[str, str]] = None  # Rename arguments before passing to API

    def __init__(
            self,
            api_key: str = None,
            api_type: str = None,
            api_version: str = None,
            api_base: str = None,

            max_retries: int = 5,
            max_calls_per_min: int = 60,
            caching: bool = True,

            model: str = None,
            temperature: float = 0.0,

            organization: Optional[str] = None,
            project: Optional[str] = None,

            rest_call: bool = False
    ):
        super().__init__()

        self.api_key = api_key
        self.api_type = api_type
        self.api_version = api_version
        self.api_base = api_base

        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        self.caching = caching

        self.model_name = model
        self.temperature = temperature

        self.organization = organization
        self.project = project

        self.rest_call = rest_call

        self.call_history = collections.deque()
        self.current_time = time.time()

        self.caller: Callable
        self._rest_headers = dict()
        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers.update({
                "Content-Type": "application/json"
            })

    @abstractmethod
    async def _library_call(self, **kwargs):
        call_args = self.parse_call_arguments(kwargs)
        pass

    @abstractmethod
    async def _rest_call(self, **kwargs):
        call_args = self.parse_call_arguments(kwargs)
        pass

    def role_start(self, role_name, **kwargs):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>" + role_name + "".join([f' {k}="{v}"' for k, v in kwargs.items()]) + "\n"

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"

    def end_of_text(self):
        return "<|endoftext|>"

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    def parse_call_arguments(self, call_args: Dict) -> Dict:
        call_exclude_arguments = self._api_exclude_arguments or {}
        call_rename_arguments = self._api_rename_arguments or {}
        parsed_call_args = {
            call_rename_arguments.get(k, k): v for k, v in call_args.items()
            if k not in call_exclude_arguments and v is not None
        }
        return parsed_call_args


class APILLMSession(LLMSession):
    def __init__(self, llm: APILLM) -> None:
        super().__init__(llm=llm)

    async def __call__(
            self,
            prompt,

            temperature=None,

            stream=None,
            caching=None,
            cache_seed: int = 0,  # TODO: Remove, not used here.
            echo: bool = False,  # TODO: Remove, not used here.
            function_call = None,  # TODO: Implement or deprecate

            n=1,
            logprobs=None,
            **call_kwargs
    ):
        # TODO: Support stream
        stream=False

        # Set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # Define the key for the cache
        cache_params = self._cache_params(locals().copy())
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1

        # check the cache
        if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):
            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            # TODO: Add tools
            # tools = extract_tools_defs(prompt)

            fail_count = 0
            error_msg = None
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "n": n,
                        "logprobs": logprobs,
                        "stream": stream,
                        **call_kwargs
                    }
                    call_out = await self.llm.caller(**call_args)

                except () as err:
                    await asyncio.sleep(3)
                    error_msg = err.message
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) {self.llm.llm_name} API errors in a row! \n"
                        f"Last error message: {error_msg}")

            if stream:
                # TODO: Support stream
                raise NotImplementedError
            else:
                llm_cache[key] = call_out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
