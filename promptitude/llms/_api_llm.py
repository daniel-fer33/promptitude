import copy
from typing import Any, Callable, Dict, List, Optional, Deque

from abc import abstractmethod
import time
import collections
import inspect
import asyncio

import pyparsing as pp

from ._llm import LLM, LLMSession


class APILLM(LLM):
    """Abstract base class for API-based LLMs"""
    _api_exclude_arguments: Optional[List[str]] = None  # Exclude arguments to pass to the API
    _api_rename_arguments: Optional[Dict[str, str]] = None  # Rename arguments before passing to API

    # Define grammar
    role_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|im_start|>"))
    role_start_name = pp.Word(pp.alphanums + "_")("role_name")
    role_kwargs = pp.Suppress(pp.Optional(" ")) + pp.Dict(pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"')))("kwargs")
    role_start = (role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")).leave_whitespace()
    role_end = pp.Suppress(pp.Literal("<|im_end|>"))
    role_content = pp.Combine(pp.ZeroOrMore(pp.CharsNotIn("<") | pp.Literal("<") + ~pp.FollowedBy("|im_end|>")))("role_content")
    role_group = pp.Group(role_start + role_content + role_end)("role_group").leave_whitespace()
    partial_role_group = pp.Group(role_start + role_content)("role_group").leave_whitespace()
    roles_grammar = pp.ZeroOrMore(role_group) + pp.Optional(partial_role_group) + pp.StringEnd()

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_type: Optional[str] = None,
            api_version: Optional[str] = None,
            api_base: Optional[str] = None,

            max_retries: int = 5,
            max_calls_per_min: int = 60,
            caching: bool = True,

            model: Optional[str] = None,
            temperature: float = 0.0,

            organization: Optional[str] = None,
            project: Optional[str] = None,

            rest_call: bool = False
    ):
        super().__init__()

        self.api_key: Optional[str] = api_key
        self.api_type: Optional[str] = api_type
        self.api_version: Optional[str] = api_version
        self.api_base: Optional[str] = api_base

        self.max_retries: int = max_retries
        self.max_calls_per_min: int = max_calls_per_min
        self.caching: bool = caching

        self.model_name: Optional[str] = model
        self.temperature: float = temperature

        self.organization: Optional[str] = organization
        self.project: Optional[str] = project

        self.rest_call: bool = rest_call

        self.call_history: Deque[float] = collections.deque()
        self.current_time: float = time.time()

        self.caller: Callable[..., Any]
        self._rest_headers: Dict[str, str] = {}
        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers.update({
                "Content-Type": "application/json"
            })

    @abstractmethod
    async def _library_call(self, **kwargs: Any) -> Any:
        """Make an API call using a library (to be implemented by subclasses)."""
        call_args = self.parse_call_arguments(kwargs)
        pass

    @abstractmethod
    async def _rest_call(self, **kwargs: Any) -> Any:
        """Make an API call using a REST endpoint (to be implemented by subclasses)."""
        call_args = self.parse_call_arguments(kwargs)
        pass

    def role_start(self, role_name: str, **kwargs: Any) -> str:
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>" + role_name + "".join([f' {k}="{v}"' for k, v in kwargs.items()]) + "\n"

    def role_end(self, role: Optional[str] = None) -> str:
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"

    def end_of_text(self) -> str:
        return "<|endoftext|>"

    def prompt_to_messages(self, prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        assert prompt.endswith(
            "<|im_start|>assistant\n"), "When calling OpenAI chat models you must generate only directly inside the assistant role! The OpenAI API does not currently support partial assistant prompting."

        parsed_prompt = self.roles_grammar.parse_string(prompt)

        for role in parsed_prompt:
            if len(role["role_content"]) > 0:
                message: Dict[str, Any] = {'role': role["role_name"], 'content': role["role_content"]}
                if "kwargs" in role:
                    for k, v in role["kwargs"].items():
                        message[k] = v
                messages.append(message)

        return messages

    def add_call(self) -> None:
        """Add the current call timestamp to call history for rate limiting."""
        now: float = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    def count_calls(self) -> int:
        """Count the number of API calls in the last minute."""
        now: float = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    def parse_call_arguments(self, call_args: Dict[str, Any]) -> Dict[str, Any]:
        """Process and prepare call arguments to be passed to the API."""
        call_exclude_arguments = self._api_exclude_arguments or []
        call_rename_arguments = self._api_rename_arguments or {}
        parsed_call_args = {
            call_rename_arguments.get(k, k): v for k, v in call_args.items()
            if k not in call_exclude_arguments and v is not None
        }
        return parsed_call_args

    async def process_stream(self, gen, key, stop_regex, n):
        """Default implementation to process and cache the streamed output."""
        list_out = []
        async for chunk in gen:
            # Process the chunk as needed (e.g., accumulate text)
            list_out.append(chunk)
            yield chunk  # Yield to the caller

        # Cache the complete output
        self.cache[key] = list_out


class APILLMSession(LLMSession):
    def __init__(self, llm: APILLM) -> None:
        super().__init__(llm=llm)

    async def __call__(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            n: int = 1,
            stop_regex: Optional[str] = None,
            caching: Optional[bool] = None,
            cache_seed: int = 0,  # Used to generate cache key
            stream: Optional[bool] = None,
            logprobs: Optional[bool] = None,
            echo: bool = False,  # TODO: Remove from callers, not used here.
            function_call = None,  # TODO: Implement or deprecate
            **call_kwargs
    ) -> Any:
        """Call the LLM with the given prompt and parameters."""

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1!"

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
                # Process the streamed output and cache it
                return self.llm.process_stream(call_out, key, stop_regex, n)
            else:
                llm_cache[key] = call_out

        # Wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]

