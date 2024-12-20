import unittest
import threading
import inspect
from promptitude.llms._llm import (
    LLM,
    LLMMeta,
    LLMSession,
    SyncSession,
    CallableAnswer,
    DiskCache,
)
from types import TracebackType
from typing import Optional, Type, Any, Dict


class DummyLLM(LLM):
    """A dummy LLM subclass for testing purposes."""
    llm_name = "dummy"

    def __init__(self, model_name="dummy-model"):
        super().__init__()
        self.model_name = model_name

    def __call__(self, *args, **kwargs):
        return "This is a test response."

    def encode(self, string: str, **kwargs):
        return [ord(c) for c in string]

    def decode(self, tokens: list, **kwargs):
        return ''.join([chr(t) for t in tokens])


class TestLLM(unittest.TestCase):
    def test_llm_initialization(self):
        """Test that the LLM base class initializes correctly."""
        llm = DummyLLM()
        self.assertEqual(llm.llm_name, "dummy")
        self.assertEqual(llm.model_name, "dummy-model")
        self.assertFalse(llm.chat_mode)

    def test_llm_call(self):
        """Test the LLM call method."""
        llm = DummyLLM()
        response = llm("Test prompt")
        self.assertEqual(response, "This is a test response.")

    def test_llm_encode_decode(self):
        """Test the encode and decode methods."""
        llm = DummyLLM()
        text = "Hello"
        tokens = llm.encode(text)
        self.assertEqual(tokens, [72, 101, 108, 108, 111])
        decoded_text = llm.decode(tokens)
        self.assertEqual(decoded_text, text)

    def test_llm_cache_sharing(self):
        """Test that all instances share the same cache."""
        llm1 = DummyLLM()
        llm2 = DummyLLM()

        # Access the cache to ensure it's initialized
        _ = llm1.cache
        _ = llm2.cache

        self.assertIs(llm1.cache, llm2.cache)
        self.assertIsInstance(llm1.cache, DiskCache)

    def test_llm_instance_cache(self):
        """Test setting and getting cache on an instance."""
        llm = DummyLLM()
        custom_cache = DiskCache("custom_cache")
        llm.cache = custom_cache
        self.assertIs(llm.cache, custom_cache)

    def test_llm_class_cache(self):
        """Test setting and getting cache on the class."""
        custom_cache = DiskCache("class_cache")
        DummyLLM.cache = custom_cache
        llm = DummyLLM()
        self.assertIs(llm.cache, custom_cache)

    def test_llm_session_asynchronous(self):
        """Test creating an asynchronous session."""
        llm = DummyLLM()
        session = llm.session(asynchronous=True)
        self.assertIsInstance(session, LLMSession)
        self.assertIs(session.llm, llm)

    def test_llm_session_synchronous(self):
        """Test creating a synchronous session."""
        llm = DummyLLM()
        session = llm.session(asynchronous=False)
        self.assertIsInstance(session, SyncSession)
        self.assertIsInstance(session._session, LLMSession)
        self.assertIs(session._session.llm, llm)

    def test_llm_session_context_manager(self):
        """Test that the session works as a context manager."""
        llm = DummyLLM()
        with llm.session(asynchronous=True) as session:
            self.assertIsInstance(session, LLMSession)

    def test_callable_answer_init(self):
        """Test initialization of CallableAnswer."""
        name = "test_function"
        args_string = '{"arg1": "value1", "arg2": 2}'
        answer = CallableAnswer(name, args_string)
        self.assertEqual(answer.__name__, name)
        self.assertEqual(answer.args_string, args_string)
        self.assertEqual(answer.__kwdefaults__, {"arg1": "value1", "arg2": 2})

    def test_callable_answer_call_without_function(self):
        """Test calling CallableAnswer without a function raises NotImplementedError."""
        answer = CallableAnswer("test_function", "{}")
        with self.assertRaises(NotImplementedError):
            answer()

    def test_callable_answer_call_with_function(self):
        """Test calling CallableAnswer with a function."""

        def test_function(arg1, arg2):
            return f"Arguments received: {arg1}, {arg2}"

        args_string = '{"arg1": "value1", "arg2": 2}'
        answer = CallableAnswer("test_function", args_string, function=test_function)
        result = answer()
        self.assertEqual(result, "Arguments received: value1, 2")

    def test_llm_serialize(self):
        """Test serialization of LLM instance."""
        llm = DummyLLM()
        serialized = llm.serialize()
        self.assertIsInstance(serialized, dict)
        self.assertIn('module_name', serialized)
        self.assertIn('class_name', serialized)
        self.assertIn('init_args', serialized)
        self.assertEqual(serialized['class_name'], 'DummyLLM')
        self.assertEqual(serialized['init_args'], {'model_name': 'dummy-model'})

    def test_llm_deserialize(self):
        """Test deserialization of LLM instance."""
        llm = DummyLLM(model_name="test-model")
        serialized = llm.serialize()

        module_name = serialized['module_name']
        class_name = serialized['class_name']
        init_args = serialized['init_args']

        # Dynamically import the class
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)

        # Create a new instance
        new_llm = cls(**init_args)
        self.assertIsInstance(new_llm, DummyLLM)
        self.assertEqual(new_llm.model_name, "test-model")

    def test_llm_meta_thread_safety(self):
        """Test that LLMMeta cache initialization is thread-safe."""

        class ThreadSafeLLM(LLM, metaclass=LLMMeta):
            llm_name = "threadsafe"

            def __init__(self, model_name="threadsafe-model"):
                super().__init__()
                self.model_name = model_name

            def __call__(self, *args, **kwargs):
                return "Thread-safe test response."

            def encode(self, string: str, **kwargs):
                return [ord(c) for c in string]

            def decode(self, tokens: list, **kwargs):
                return ''.join([chr(t) for t in tokens])

        def access_cache():
            llm = ThreadSafeLLM()
            _ = llm.cache

        threads = []
        for _ in range(10):
            t = threading.Thread(target=access_cache)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        llm = ThreadSafeLLM()
        self.assertIsInstance(llm.cache, DiskCache)

    def test_llm_excluded_args(self):
        """Test that excluded_args are not included in serialization."""

        class ExcludeArgsLLM(LLM):
            llm_name = "exclude_args_test"
            excluded_args = ['api_key', 'secret_value']

            def __init__(self, model_name="exclude-model", api_key="secret", secret_value="top_secret"):
                super().__init__()
                self.model_name = model_name
                self.api_key = api_key
                self.secret_value = secret_value

            def __call__(self, *args, **kwargs):
                return "Exclude args test response."

            def encode(self, string: str, **kwargs):
                return [ord(c) for c in string]

            def decode(self, tokens: list, **kwargs):
                return ''.join([chr(t) for t in tokens])

        llm = ExcludeArgsLLM()
        serialized = llm.serialize()
        self.assertNotIn('api_key', serialized['init_args'])
        self.assertNotIn('secret_value', serialized['init_args'])
        self.assertIn('model_name', serialized['init_args'])

    def test_llm_class_attribute_map(self):
        """Test that class_attribute_map is correctly applied in serialization."""

        class AttributeMapLLM(LLM):
            llm_name = "attribute_map_test"
            class_attribute_map = {'model': 'model_name'}

            def __init__(self, model="attribute-model"):
                super().__init__()
                self.model_name = model

            def __call__(self, *args, **kwargs):
                return "Attribute map test response."

            def encode(self, string: str, **kwargs):
                return [ord(c) for c in string]

            def decode(self, tokens: list, **kwargs):
                return ''.join([chr(t) for t in tokens])

        llm = AttributeMapLLM()
        serialized = llm.serialize()
        self.assertIn('model', serialized['init_args'])
        self.assertEqual(serialized['init_args']['model'], 'attribute-model')

    def test_sync_session_call(self):
        """Test calling the SyncSession."""
        llm = DummyLLM()

        async def async_call(*args, **kwargs):
            return llm(*args, **kwargs)

        class MockLLMSession(LLMSession):
            async def __call__(self, *args, **kwargs):
                return await async_call(*args, **kwargs)

        session = SyncSession(MockLLMSession(llm))
        result = session("Test prompt")
        self.assertEqual(result, "This is a test response.")

    def test_llm_getitem(self):
        """Test getting attribute via __getitem__."""
        llm = DummyLLM()
        self.assertEqual(llm['model_name'], 'dummy-model')
        with self.assertRaises(AttributeError):
            _ = llm['non_existent_attribute']

    def test_llm_function_call_stop_regex(self):
        """Test default function_call_stop_regex."""
        llm = DummyLLM()
        expected_regex = r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```"
        self.assertEqual(llm.function_call_stop_regex, expected_regex)
