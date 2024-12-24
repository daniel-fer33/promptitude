import time
import collections

from ._llm import LLM


class LocalLLM(LLM):
    """Abstract base class for locally hosted LLMs."""

    def __init__(
            self,
            model: str = None,
            tokenizer=None,
            device=None,
            caching: bool = True,
            temperature: float = 0.0,
            **kwargs):
        super().__init__()
        self.model_name = model if isinstance(model, str) else model.__class__.__name__

        # Load the model and tokenizer
        self.model_obj, self.tokenizer = self._load_model_and_tokenizer(model, tokenizer, **kwargs)
        if device is not None: # set the device if requested
            self.model_obj = self.model_obj.to(device)
        self.device = self.model_obj.device # otherwise note the current device

        self.caching = caching
        self.temperature = temperature

        self.current_time = time.time()
        self.call_history = collections.deque()

    def _load_model_and_tokenizer(self, model, tokenizer, **kwargs):
        """Load model and tokenizer based on the model_name."""
        raise NotImplementedError("Subclasses must implement the _load_model_and_tokenizer method.")

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)