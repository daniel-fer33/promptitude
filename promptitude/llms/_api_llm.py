from typing import List, Dict, Optional, Any

from ._llm import LLM
import time
import collections
import inspect
import asyncio


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

            deployment_id: Optional[str] = None,
            organization: Optional[str] = None,

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

        self.deployment_id = deployment_id
        self.organization = organization

        self.rest_call = rest_call

        self.call_history = collections.deque()
        self.current_time = time.time()

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

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
