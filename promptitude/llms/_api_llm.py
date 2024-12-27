import copy
from typing import Any, Callable, Dict, List, Optional, Deque, Tuple, Type, AsyncIterator

from abc import abstractmethod
import time
import collections
import inspect
import asyncio
import logging
import re

import pyparsing as pp

from ._llm import LLM, LLMSession

log = logging.getLogger(__name__)


class AsyncRateLimiter:
    def __init__(self, max_rate: float, time_period: float):
        """ Initialize the rate limiter. """
        self._max_rate = max_rate  # The maximum number of calls allowed in the time period.
        self._time_period = time_period   # The time period in seconds during which max_rate tokens can be used.
        self._calls = max_rate
        self._lock = asyncio.Lock()
        self._last_check = time.monotonic()

    async def acquire(self):
        """ Acquire permission to proceed with an API call. """
        async with self._lock:
            current_time = time.monotonic()
            elapsed = current_time - self._last_check
            self._last_check = current_time

            # Refill tokens based on elapsed time
            refill = (elapsed / self._time_period) * self._max_rate
            self._calls = min(self._calls + refill, self._max_rate)

            if self._calls >= 1:
                # Consume a call and proceed
                self._calls -= 1
            else:
                # Calculate wait time until a new token is available
                wait_time = ((1 - self._calls) * self._time_period) / self._max_rate
                await asyncio.sleep(wait_time)
                self._last_check = time.monotonic()
                self._calls = 0  # Calls just used


class APILLM(LLM):
    """Abstract base class for API-based LLMs"""
    _api_exclude_arguments: Optional[List[str]] = None  # Exclude arguments to pass to the API
    _api_rename_arguments: Optional[Dict[str, str]] = None  # Rename arguments before passing to API
    api_exceptions: Tuple[Type[BaseException], ...] = ()

    # Define grammar
    role_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|im_start|>"))
    role_start_name = pp.Word(pp.alphanums + "_")("role_name")
    role_kwargs = pp.Suppress(pp.Optional(" ")) + pp.Dict(
        pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"')))("kwargs")
    role_start = (role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")).leave_whitespace()
    role_end = pp.Suppress(pp.Literal("<|im_end|>"))
    role_content = pp.Combine(pp.ZeroOrMore(pp.CharsNotIn("<") | pp.Literal("<") + ~pp.FollowedBy("|im_end|>")))(
        "role_content")
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

        # Initialize variables
        self._api_exclude_arguments = self._api_exclude_arguments or []
        self._api_rename_arguments = self._api_rename_arguments or {}
        # Exclude 'prompt' and 'messages'
        self._api_exclude_arguments += ['prompt', 'messages']

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

        self.rate_limiter = AsyncRateLimiter(max_rate=max_calls_per_min, time_period=60.0)

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
        raise NotImplementedError("Subclasses must implement the _library_call method.")

    @abstractmethod
    async def _rest_call(self, **kwargs: Any) -> Any:
        """Make an API call using a REST endpoint (to be implemented by subclasses)."""
        call_args = self.parse_call_arguments(kwargs)
        raise NotImplementedError("Subclasses must implement the _rest_call method.")

    def role_start(self, role_name: str, **kwargs: Any) -> str:
        """Generate the role start token for the chat prompt."""
        if not self.chat_mode:
            raise ValueError("role_start() can only be used in chat mode")
        return "<|im_start|>" + role_name + "".join([f' {k}="{v}"' for k, v in kwargs.items()]) + "\n"

    def role_end(self, role: Optional[str] = None) -> str:
        """Generate the role end token for the chat prompt."""
        if not self.chat_mode:
            raise ValueError("role_end() can only be used in chat mode")
        return "<|im_end|>"

    def end_of_text(self) -> str:
        return "<|endoftext|>"

    def prompt_to_messages(self, prompt: str) -> List[Dict[str, Any]]:
        if not prompt:
            log.error("Prompt is empty or None.")
            raise ValueError("Prompt cannot be empty or None.")

        try:
            messages: List[Dict[str, Any]] = []

            if not prompt.endswith("<|im_start|>assistant\n"):
                raise ValueError(
                    "When calling chat models, you must generate only inside the assistant role! "
                    "The API does not currently support partial assistant prompting."
                )
            parsed_prompt = self.roles_grammar.parse_string(prompt)

            for role in parsed_prompt:
                if len(role["role_content"]) > 0:
                    message: Dict[str, Any] = {'role': role["role_name"], 'content': role["role_content"]}
                    if "kwargs" in role:
                        for k, v in role["kwargs"].items():
                            message[k] = v
                    messages.append(message)

            return messages
        except pp.ParseException as e:
            log.error(f"Failed to parse prompt: {e}")
            raise ValueError(f"Invalid prompt format: {e}")

    def parse_call_arguments(self, call_args: Dict[str, Any]) -> Dict[str, Any]:
        """Process and prepare call arguments to be passed to the API."""
        # Exclude 'prompt' and 'messages' from being directly passed to the API
        parsed_call_args = {
            self._api_rename_arguments.get(k, k): v for k, v in call_args.items()
            if k not in self._api_exclude_arguments and v is not None
        }

        # Handle the case when 'messages' are provided
        if 'messages' in call_args and call_args['messages'] is not None:
            parsed_call_args['messages'] = call_args['messages']
        elif 'prompt' in call_args and call_args['prompt'] is not None:
            # Convert prompt to messages
            parsed_call_args['messages'] = self.prompt_to_messages(call_args['prompt'])
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        return parsed_call_args

    async def process_stream(self, gen: AsyncIterator[str], key: Any, stop_regex: Optional[str], n: int
                             ) -> AsyncIterator[str]:
        """Default implementation to process and cache the streamed output."""
        list_out = []
        pattern = re.compile(stop_regex) if stop_regex else None
        async for chunk in gen:
            list_out.append(chunk)
            yield chunk  # Yield to the caller
            if pattern and pattern.search(chunk):
                # If stop condition met, break the loop
                break

        # Cache the complete output
        self.cache[key] = list_out


class APILLMSession(LLMSession):
    def __init__(self, llm: APILLM) -> None:
        super().__init__(llm=llm)

    async def __call__(
            self,
            prompt: Optional[str] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            temperature: Optional[float] = None,
            n: int = 1,
            stop_regex: Optional[str] = None,
            caching: Optional[bool] = None,
            cache_seed: int = 0,  # Used to generate cache key
            stream: Optional[bool] = None,
            logprobs: Optional[bool] = None,
            echo: bool = False,  # TODO: Remove from callers, not used here.
            function_call: Optional[str] = None,  # TODO: Implement or deprecate
            **call_kwargs: Any
    ) -> Any:
        """Call the LLM with the given prompt or messages and parameters."""

        if function_call not in [None, 'none']:
            raise NotImplementedError("The 'function_call' parameter is not implemented yet.")

        # Ensure that exactly one of prompt or messages is provided
        if (prompt is None and messages is None) or (prompt is not None and messages is not None):
            raise ValueError("Exactly one of 'prompt' or 'messages' must be provided.")

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        if stop_regex is not None and not stream:
            raise ValueError("We can only support stop_regex when stream=True!")
        if stop_regex is not None and n != 1:
            raise ValueError("We don't yet support stop_regex combined with n > 1!")

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
            await self.llm.rate_limiter.acquire()

            # TODO: Add tools
            # tools = extract_tools_defs(prompt)

            fail_count = 0
            backoff = 1  # Start with a 1-second backoff
            max_backoff = 60  # Set a maximum backoff time
            error_msg = None
            while True:
                should_retry = False
                try:
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "messages": messages,
                        "temperature": temperature,
                        "n": n,
                        "logprobs": logprobs,
                        "stream": stream,
                        **call_kwargs
                    }
                    call_out = await self.llm.caller(**call_args)

                except self.llm.api_exceptions as err:
                    error_msg = str(err)
                    fail_count += 1
                    if fail_count >= self.llm.max_retries:
                        log.error(
                            f"Exceeded maximum retries ({self.llm.max_retries}) for {self.llm.llm_name} API. "
                            f"Last error: {error_msg}"
                        )
                        raise APIRateLimitException(
                            f"Too many (more than {self.llm.max_retries}) {self.llm.llm_name} API errors in a row! \n"
                            f"Last error message: {error_msg}"
                        ) from err
                    else:
                        log.warning(
                            f"API call failed for model {self.llm.model_name} with error: {error_msg}. "
                            f"Retrying in {backoff} seconds..."
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)  # Exponential backoff with a cap
                        should_retry = True

                if not should_retry:
                    break

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


class APILLMException(Exception):
    """Base exception class for APILLM."""


class APIRateLimitException(APILLMException):
    """Exception raised when API rate limit is exceeded."""
