from typing import Any, Optional, Union, List, Dict

from ._llm import LLM


class Mock(LLM):
    """Mock class for testing."""

    def __init__(self, output: Optional[Union[str, List[Any], Dict[str, Any]]] = None) -> None:
        """ Initialize the mock class.
        
        Parameters
        ----------
        output : str or list or dict
            The output of the mock class. If a list is provided, the mock
            class will return the next item in the list each time it is
            called. Otherwise, the same output will be returned each time.
            If a dictionary is provided, the mock class will choose the first
            dictionary key that matches a suffix of the input prompt, and use
            the string or list value associated with that key for generation.
        """
        super().__init__()

        # Ensure the output is always a dictionary of lists
        if output is None:
            output = {"": [f"mock output {i}" for i in range(100)]}
        if isinstance(output, str):
            output = [output]
        if isinstance(output, list):
            output = {"": output}
        for key in output.keys():
            if not isinstance(output[key], list):
                output[key] = [output[key]]

        self.output: Dict[str, List[Any]] = output
        self.counts: Dict[str, int] = {k: 0 for k in output.keys()}
        self._sorted_keys: List[str] = sorted(self.output.keys(), key=lambda k: len(k), reverse=True)
        self._tokenizer = MockTokenizer()

    def __call__(
        self,
        prompt: str,
        *args: Any,
        n: int = 1,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        key = self._find_suffix_match(prompt)
        output_list = self.output[key]
        choices = []
        for i in range(n):
            out = output_list[min(self.counts[key], len(output_list) - 1)]
            self.counts[key] += 1
            if isinstance(out, str):
                choices.append({"text": out, "finish_reason": "stop"})
            elif isinstance(out, dict):
                choices.append(out)
            else:
                raise ValueError(f"Invalid output type: {type(out)}")

        out = {"choices": choices}

        if stream:
            return [out]
        else:
            return out

    def role_start(self, role_name: str, **kwargs: Any) -> str:
        attributes = "".join([f' {k}="{v}"' for k, v in kwargs.items()])
        return f"<|im_start|>{role_name}{attributes}\n"

    def role_end(self, role_name: Optional[str] = None) -> str:
        return "<|im_end|>"

    def encode(self, string: str, **kwargs: Any) -> List[int]:
        return self._tokenizer.encode(string)

    def decode(self, tokens: List[int], **kwargs: Any) -> str:
        return self._tokenizer.decode(tokens)

    def _find_suffix_match(self, prompt: str) -> str:
        """Find the output key that matches the longest suffix of the prompt."""
        for key in self._sorted_keys:
            if prompt.endswith(key):
                return key
        return ""


class MockTokenizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode(text: str) -> List[int]:
        return [s for s in text.encode("utf-8")]

    @staticmethod
    def decode(ids: List[int]) -> str:
        return "".join([chr(i) for i in ids])
