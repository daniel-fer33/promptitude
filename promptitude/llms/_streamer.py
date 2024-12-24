class BaseStreamer:
    def __init__(self, stop_regex=None):
        self.stop_regex = stop_regex
        # Common initialization

    def process_stream_chunk(self, chunk):
        """Process a chunk of the stream."""
        raise NotImplementedError("Subclasses must implement the process_stream_chunk method.")