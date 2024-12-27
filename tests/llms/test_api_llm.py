import unittest
import asyncio
import time
import re
from typing import Optional, Type, Any, Dict, List, AsyncIterator
import unittest.mock as mock

from promptitude.llms._api_llm import (
    AsyncRateLimiter,
    APILLM,
    APILLMSession,
    APILLMException,
    APIRateLimitException,
)


class DummyAPILLM(APILLM):
    """A dummy API-based LLM subclass for testing purposes."""

    def __init__(self, model_name="dummy-api-model", *args, **kwargs):
        super().__init__(model=model_name, *args, **kwargs)
        self.model_name = model_name
        self.llm_name = "dummy_api_llm"
        self.api_exceptions = (APIRateLimitException,)

    async def _library_call(self, **kwargs):
        # Simulate a response
        return {"choices": [{"text": "This is a library call test response."}]}

    async def _rest_call(self, **kwargs):
        # Simulate a response
        return {"choices": [{"text": "This is a REST call test response."}]}

    async def process_stream(self, gen: AsyncIterator[str], key: Any, stop_regex: Optional[str], n: int):
        # For testing, simply collect all chunks and cache them
        chunks = []
        async for chunk in gen:
            if stop_regex and re.search(stop_regex, chunk):
                chunks.append(chunk)
                break
            chunks.append(chunk)
            yield chunk
        self.cache[key] = chunks

    def encode(self, string: str, **kwargs):
        return [ord(c) for c in string]

    def decode(self, tokens: list, **kwargs):
        return ''.join([chr(t) for t in tokens])


class TestAsyncRateLimiter(unittest.IsolatedAsyncioTestCase):

    async def test_rate_limiter_allows_calls_within_rate(self):
        rate_limiter = AsyncRateLimiter(max_rate=5, time_period=1)
        times = []

        for _ in range(5):
            await rate_limiter.acquire()
            times.append(time.monotonic())

        # All calls should have been allowed without significant delay
        intervals = [t2 - t1 for t1, t2 in zip(times, times[1:])]
        self.assertTrue(all(interval < 0.05 for interval in intervals))

    async def test_rate_limiter_blocks_calls_exceeding_rate(self):
        rate_limiter = AsyncRateLimiter(max_rate=2, time_period=1)
        times = []

        for _ in range(3):
            await rate_limiter.acquire()
            times.append(time.monotonic())

        total_time = times[-1] - times[0]
        # The third call should be delayed by about 0.5 seconds (since we are allowed 2 calls per second)
        self.assertTrue(total_time >= 0.5, f"Total time {total_time} is less than expected")


class TestAPILLMSession(unittest.IsolatedAsyncioTestCase):

    async def test_api_session_call_successful(self):
        llm = DummyAPILLM()
        llm.cache.clear()  # Clear the cache to avoid interference from previous tests
        session = APILLMSession(llm)

        result = await session(prompt="<|im_start|>user\nHello!<|im_end|><|im_start|>assistant\n")
        self.assertIsNotNone(result)
        # Since we didn't specify a cache, the result comes from llm.caller
        expected_result = {"choices": [{"text": "This is a library call test response."}]}
        self.assertEqual(result, expected_result)

    @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
    async def test_api_session_call_with_retry(self, mock_sleep):
        llm = DummyAPILLM()
        llm.cache.clear()  # Clear the cache to avoid interference from previous tests
        session = APILLMSession(llm)

        # Mock the llm.caller to raise API exceptions and then eventually succeed
        llm.caller = mock.AsyncMock(side_effect=[
            APIRateLimitException("Rate limit exceeded"),
            APIRateLimitException("Rate limit exceeded"),
            {"choices": [{"text": "Successful response after retries."}]}
        ])

        llm.max_retries = 3  # Set max retries for the test

        result = await session(prompt="<|im_start|>user\nHello!<|im_end|><|im_start|>assistant\n")
        self.assertEqual(result, {"choices": [{"text": "Successful response after retries."}]})

        # Ensure that the sleep was called the expected number of times (max_retries - 1 times)
        self.assertEqual(mock_sleep.await_count, 2)
