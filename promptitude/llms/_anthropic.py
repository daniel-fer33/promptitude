from typing import List, Dict, Optional, Tuple, Type

import os
import copy
import asyncio
import types
import regex
import logging
import pyparsing as pp

import anthropic
from anthropic import AsyncAnthropic, AsyncStream

from ._llm import LLMSession, SyncSession
from ._api_llm import APILLM, APILLMSession
from ._streamer import BaseStreamer

log = logging.getLogger(__name__)


def anthropic_messages_response_to_opeanai_completion_dict(messages_response):
    out = messages_response.model_dump()
    if 'message' in out:
        # Stream
        message_type = out['type']
        out = out['message']
        out['type'] = message_type

    finish_reason = {
        'end_turn': 'stop',
        'max_tokens': 'length',
        'stop_sequence': 'stop'
    }.get(out.get('stop_reason', out.get('delta', {}).get('stop_reason', None)), None)
    if 'content' in out:
        out['choices'] = [
            dict(
                finish_reason=finish_reason,
                index=i,
                message=dict(
                    content=s1['text'],
                    role=out['role'],
                    function_call=None,  # TODO: Implement
                    tool_calls=None  # TODO: Implement
                ),
                logprobs=None
            ) for i, s1 in enumerate(out['content'])
        ]

    # Remove keys
    remove_keys = ['content', 'role', 'stop_reason', 'stop_sequence']
    for k in remove_keys:
        if k in out:
            del out[k]
    return out


async def add_text_to_chat_mode_generator(chat_mode_gen):
    delta_out = None
    async for event in chat_mode_gen:
        event_type = event.type
        #log.debug(f"[add_text_to_chat_mode_generator | resp]: {event_type} {event.dict()}\n")
        resp = anthropic_messages_response_to_opeanai_completion_dict(event)

        if event_type == 'message_start':
            delta_out = resp
        elif event_type == 'content_block_start':
            delta_out['choices'].append({"text": ''})
        elif event_type == 'content_block_delta':
            delta_out['choices'][resp['index']]['text'] = resp['delta']['text']
            yield delta_out
        elif event_type == 'content_block_stop':
            pass
        elif event_type == 'message_delta':
            delta_out['choices'][-1]['text'] = ''
            if 'finish_reason' in resp:
                delta_out['finish_reason'] = resp['finish_reason']
            if ('usage' in delta_out and 'output_tokens' in delta_out['usage'] and
                    'usage' in resp and 'output_tokens' in resp['usage']):
                delta_out['usage']['output_tokens'] += resp['usage']['output_tokens']
            yield delta_out
        elif event_type == 'message_stop':
            delta_out['type'] = 'message_stop'
            # break
        else:
            raise ValueError(f"Stream event not found: {event_type} {event.dict()}")


def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, (types.AsyncGeneratorType, types.GeneratorType, AsyncStream)):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        chat_mode = anthropic_messages_response_to_opeanai_completion_dict(chat_mode)
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
        return chat_mode


def merge_stream_chunks(first_chunk, second_chunk):
    """ This merges two stream responses together.
    """

    out = copy.deepcopy(first_chunk)

    # merge the choices
    for i in range(len(out['choices'])):
        out_choice = out['choices'][i]
        second_choice = second_chunk['choices'][i]
        out_choice['text'] += second_choice['text']
        if 'index' in second_choice:
            out_choice['index'] = second_choice['index']
        if 'finish_reason' in second_choice:
            out_choice['finish_reason'] = second_choice['finish_reason']
        if out_choice.get('logprobs', None) is not None:
            out_choice['logprobs']['token_logprobs'] += second_choice['logprobs']['token_logprobs']
            out_choice['logprobs']['top_logprobs'] += second_choice['logprobs']['top_logprobs']
            out_choice['logprobs']['text_offset'] = second_choice['logprobs']['text_offset']

    return out


class Anthropic(APILLM):
    llm_name: str = "anthropic"

    # API
    _api_exclude_arguments: Optional[List[str]] = [
        'organization', 'n', 'prompt'
    ]
    _api_rename_arguments: Optional[Dict[str, str]] = {
        'stop': 'stop_sequences'
    }
    api_exceptions: Tuple[Type[BaseException], ...] = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        # anthropic.APIError,
        anthropic.APITimeoutError
    )

    # Serialization
    excluded_args: List[str] = ['api_key', 'api_type']
    class_attribute_map: Dict[str, str] = {
        'model': 'model_name',
    }

    def __init__(
            self,
            model=None,
            api_key=None,
            **kwargs
    ):

        api_type = "anthropic"

        # fill in default API key value
        if api_key is None:  # get from environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is not None and not api_key.startswith("sk-ant-") and os.path.exists(
                os.path.expanduser(api_key)):  # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None:  # get from default file location
            try:
                with open(os.path.expanduser('~/.anthropic_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")

        # fill in default model value
        if model is None:
            model = os.environ.get("ANTHROPIC_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.anthropic_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        super().__init__(
            model=model, api_key=api_key, api_type=api_type, **kwargs
        )

        # Tokenizer
        self._tokenizer = anthropic.Client().beta.messages.count_tokens

        self.chat_mode = True  # Only Anthropic chat-mode will be supported

    def session(self, asynchronous=False):
        if asynchronous:
            return APILLMSession(self)
        else:
            return SyncSession(APILLMSession(self))

    @classmethod
    async def process_stream(cls, gen, key, stop_regex, n):
        assert not stop_regex, "Currently `stop_regex` is not implemented for Anthropic API"

        list_out = []
        cached_out = None

        # init stop_regex variables
        if stop_regex is not None:
            if isinstance(stop_regex, str):
                stop_patterns = [regex.compile(stop_regex)]
            else:
                stop_patterns = [regex.compile(pattern) for pattern in stop_regex]

            current_strings = ["" for _ in range(n)]
            # last_out_pos = ["" for _ in range(n)]

        # iterate through the stream
        all_done = False
        async for curr_out in gen:

            # if we have a cached output, extend it with the current output
            if cached_out is not None:
                out = merge_stream_chunks(cached_out, curr_out)
            else:
                out = curr_out

            # check if we have stop_regex matches
            found_partial = False
            if stop_regex is not None:

                # keep track of the generated text so far
                for i, choice in enumerate(curr_out['choices']):
                    current_strings[i] += choice['text']

                # check if all of the strings match a stop string (and hence we can stop the batch inference)
                all_done = True
                for i in range(len(current_strings)):
                    found = False
                    for s in stop_patterns:
                        if s.search(current_strings[i]):
                            found = True
                    if not found:
                        all_done = False
                        break

                # find where trim off the stop regex matches if needed (and look for partial matches)
                stop_pos = [1e10 for _ in range(n)]
                stop_text = [None for _ in range(n)]
                for i in range(len(current_strings)):
                    for s in stop_patterns:
                        m = s.search(current_strings[i], partial=True)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if m.partial:  # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_text[i] = current_strings[i][span[0]:span[1]]
                                    stop_pos[i] = min(span[0], stop_pos[i])
                    if stop_pos != 1e10:
                        stop_pos[i] = stop_pos[i] - len(current_strings[i])  # convert to relative position from the end

            # if we might be starting a stop sequence, we need to cache the output and continue to wait and see
            if found_partial:
                cached_out = out
                continue

            # if we get here, we are not starting a stop sequence, so we can emit the output
            else:
                cached_out = None

                if stop_regex is not None:
                    for i in range(len(out['choices'])):
                        if stop_pos[i] < len(out['choices'][i]['text']):
                            out['choices'][i] = out['choices'][
                                i].to_dict()  # because sometimes we might need to set the text to the empty string (and OpenAI's object does not like that)
                            out['choices'][i]['text'] = out['choices'][i]['text'][:stop_pos[i]]
                            out['choices'][i]['stop_text'] = stop_text[i]
                            out['choices'][i]['finish_reason'] = "stop"

                list_out.append(out)
                yield out
                if all_done or out.get('type', None) == 'message_stop':
                    #gen.close()
                    break

        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out

    async def _library_call(self, **call_kwargs):
        """ Call the Anthropic API using the python package."""
        assert self.api_key is not None, "You must provide an Anthropic API key to use the Anthropic LLM. " \
                                         "Either pass it in the constructor, set the ANTHROPIC_API_KEY environment " \
                                         "variable, or create the file ~/.anthropic_api_key with your key in it."

        # Filter non supported call arguments
        n = call_kwargs['n']
        if n > 1:
            raise ValueError(f"Anthropic doesn't support more than one chat completion (n>1={n})")
        assert 'n' in self._api_exclude_arguments

        # Handle messages
        if 'messages' in call_kwargs and call_kwargs['messages'] is not None:
            messages = call_kwargs['messages']
        elif 'prompt' in call_kwargs and call_kwargs['prompt'] is not None:
            messages = self.prompt_to_messages(call_kwargs['prompt'])
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        system_msgs = [s1['content'] for s1 in messages if s1['role'] == 'system']
        call_kwargs['system'] = system_msgs[-1] if len(system_msgs) > 0 else None
        call_kwargs['messages'] = [s1 for s1 in messages if s1['role'] != 'system']

        # Remove 'prompt' from call_kwargs if present
        call_kwargs.pop('prompt', None)
        assert 'prompt' in self._api_exclude_arguments

        # Parse call arguments
        call_args = self.parse_call_arguments(call_kwargs)
        if 'stop_sequences' in call_args and not isinstance(call_args['stop_sequences'], list):
            # stop_sequences: Input should be a valid list
            call_args['stop_sequences'] = [call_args['stop_sequences'], ]

        # Start API client
        client = AsyncAnthropic(api_key=self.api_key)

        # Call LLM API
        out = await client.messages.create(**call_args)
        log.info(f"LLM call response: {out}")
        out = add_text_to_chat_mode(out)

        return out

    async def _rest_call(self, **kwargs):
        raise NotImplementedError

    async def _rest_stream_handler(self, responses):
        raise NotImplementedError

    def encode(self, string: str, **kwargs) -> List[int]:
        # note that is_fragment is not used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self._allowed_special_tokens, **kwargs)

    def decode(self, tokens: List[int], **kwargs) -> str:
        return self._tokenizer.decode(tokens, **kwargs)


class AnthropicSession(APILLMSession):
    pass


class AnthropicStreamer(BaseStreamer):
    def process_stream_chunk(self, chunk):
        raise NotImplementedError


class _AnthropicSession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, top_k=None, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, function_call=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the Anthropic API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the Anthropic API!"

        assert token_healing is None or token_healing is False, "The Anthropic API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The Anthropic API does not support Guidance pattern controls! " \
                            "Please either switch to an endpoint that does, or don't use the `pattern` " \
                            "argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
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
                        "max_tokens": max_tokens,
                        "stop_sequences": stop,
                        "stream": stream,
                        "temperature": temperature,
                        #"tools": tools,
                        "top_p": top_p,
                        "top_k": top_k,
                        **completion_kwargs
                    }

                    call_args = {k: v for k, v in call_args.items() if v is not None}
                    out = await self.llm.caller(**call_args)

                except (anthropic.RateLimitError,
                        anthropic.APIConnectionError,
                        anthropic.APIStatusError,
                        #anthropic.APIError,
                        anthropic.APITimeoutError) as err:
                    await asyncio.sleep(3)
                    error_msg = err.message
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) Anthropic API errors in a row! \n"
                        f"Last error message: {error_msg}")

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                llm_cache[key] = out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
