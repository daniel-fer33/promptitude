from typing import List, Dict, Optional

import os
import copy
import asyncio
import types
import regex
import logging
import pyparsing as pp

import openai
from openai import AsyncOpenAI, AsyncStream

from ._llm import LLMSession, SyncSession
from ._api_llm import APILLM, APILLMSession
from ._streamer import BaseStreamer

log = logging.getLogger(__name__)

# After the changes introduced in PR #1488 (https://github.com/openai/openai-python/pull/1488),
# in some cases, pytest is not properly finalizing after all tests pass successfully.
# This is a temporary of the issue
from openai._base_client import get_platform
PLATFORM = get_platform()


# define the syntax for the function definitions
start_functions = pp.Suppress(pp.Literal("## functions\n\nnamespace functions {\n\n"))
comment = pp.Combine(pp.Suppress(pp.Literal("//") + pp.Optional(" ")) + pp.restOfLine)
end_functions = pp.Suppress("} // namespace functions")
function_def_start = pp.Optional(comment)("function_description") + pp.Suppress(pp.Literal("type")) + pp.Word(pp.alphas + "_")("function_name") + pp.Suppress(pp.Literal("=") + pp.Literal("(_:") + pp.Literal("{"))
function_def_end = pp.Suppress(pp.Literal("})") + pp.Literal("=>") + pp.Literal("any;"))
parameter_type = (pp.Word(pp.alphas + "_")("simple_type") | pp.QuotedString('"')("enum_option") + pp.OneOrMore(pp.Suppress("|") + pp.QuotedString('"')("enum_option"))("enum")) + pp.Suppress(pp.Optional(","))
parameter_def = pp.Optional(comment)("parameter_description") + pp.Word(pp.alphas + "_")("parameter_name") + pp.Optional(pp.Literal("?"))("is_optional") + pp.Suppress(pp.Literal(":")) + pp.Group(parameter_type)("parameter_type")
function_def = function_def_start + pp.OneOrMore(pp.Group(parameter_def)("parameter")) + function_def_end
functions_def = start_functions + pp.OneOrMore(pp.Group(function_def)("function")) + end_functions


async def add_text_to_chat_mode_generator(chat_mode):
    in_function_call = False
    async for part in chat_mode:
        resp = part.model_dump()
        if "choices" in resp:
            for c in resp['choices']:

                # move content from delta to text so we have a consistent interface with non-chat mode
                found_content = False
                if "content" in c['delta']:
                    if c['delta']['content'] is None:
                        c['delta']['content'] = ""
                    if c['delta']['content'] != "":
                        found_content = True
                        c['text'] = c['delta']['content']

                # capture function call data and convert to text again so we have a consistent interface with non-chat mode and open models
                if "function_call" in c['delta'] and c['delta']['function_call'] is not None:

                    # build the start of the function call (the follows the syntax that GPT says it wants when we ask it, and will be parsed by the @function_detector)
                    if not in_function_call:
                        start_val = "\n```typescript\nfunctions." + c['delta']['function_call']["name"] + "("
                        if not c['text']:
                            c['text'] = start_val
                        else:
                            c['text'] += start_val
                        in_function_call = True

                    # extend the arguments JSON string
                    val = c['delta']['function_call']["arguments"]
                    if 'text' in c:
                        c['text'] += val
                    else:
                        c['text'] = val

                if 'logprobs' in c and c['logprobs'] is not None:
                    if 'top_logprobs' not in c['logprobs']:
                        c['logprobs']['top_logprobs'] = {}  # TODO: This probably has to be a list
                    c['logprobs']['top_logprobs'].update({s1['token']: s1['logprob'] for s1 in c['logprobs']['content']})
                if not found_content and not in_function_call:
                    break  # the role markers are outside the generation in chat mode right now TODO: consider how this changes for uncontrained generation
            else:
                yield resp
        else:
            yield resp

    # close the function call if needed
    if in_function_call:
        yield {'choices': [{'text': ')```'}]}


def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, (types.AsyncGeneratorType, types.GeneratorType, AsyncStream)):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        chat_mode = chat_mode.model_dump()
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
            if 'logprobs' in c and c['logprobs'] is not None:
                c['logprobs']['top_logprobs'] = {s1['token']: s1['logprob'] for s1 in c['logprobs']['content']}
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


def get_json_from_parse(parse_out):
    functions = []
    for function in parse_out:
        function_name = function.function_name
        function_description = function.function_description
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        for parameter in function:
            if isinstance(parameter, str):
                continue
            parameter_name = parameter.parameter_name
            parameter_description = parameter.parameter_description
            parameter_type = parameter.parameter_type
            is_optional = parameter.is_optional
            d = {}
            if parameter_type.simple_type:
                d["type"] = parameter_type.simple_type
            elif parameter_type.enum:
                d["type"] = "string"
                d["enum"] = [s for s in parameter_type]
            if parameter_description:
                d["description"] = parameter_description
            if not is_optional:
                parameters["required"].append(parameter_name)
            parameters["properties"][parameter_name] = d
        functions.append({
            "name": function_name,
            "description": function_description,
            "parameters": parameters
        })
    return functions


def extract_function_defs(prompt):
    """ This extracts function definitions from the prompt.
    """

    if "\n## functions\n" not in prompt:
        return None
    else:
        functions_text = prompt[prompt.index("\n## functions\n")+1:prompt.index("} // namespace functions")+24]
        parse_out = functions_def.parseString(functions_text)
        return get_json_from_parse(parse_out)


class OpenAI(APILLM):
    llm_name: str = "openai"
    chat_model_pattern: str = r'^(gpt-3\.5-turbo|gpt-4|gpt-4-vision|gpt-4-turbo|gpt-4o|gpt-4o-mini|o1-preview|o1-mini)(-\d+k)?(-\d{4})?(-vision)?(-instruct)?(-\d{2})?(-\d{2})?(-preview)?$'
    default_allowed_special_tokens: List[str] = ["<|endoftext|>", "<|endofprompt|>"]

    # API
    _api_exclude_arguments: Optional[List[str]] = [
        'prompt'
    ]
    _api_rename_arguments: Optional[Dict[str, str]] = {}

    # Serialization
    excluded_args: List[str] = ['api_key', 'api_type']
    class_attribute_map: Dict[str, str] = {
        'model': 'model_name',
        'encoding_name': '_encoding_name',
        'allowed_special_tokens': '_allowed_special_tokens'
    }

    def __init__(
            self,
            model=None,
            api_key=None,
            allowed_special_tokens: Optional[str] = None,
            encoding_name: Optional[str] = None,
            **kwargs
    ):
        api_type = "open_ai"

        # fill in default API key value
        if api_key is None:  # get from environment variable
            api_key = os.environ.get("OPENAI_API_KEY", getattr(openai, "api_key", None))
        if api_key is not None and not api_key.startswith("sk-") and os.path.exists(
                os.path.expanduser(api_key)):  # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None:  # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")

        # fill in default model value
        if model is None:
            model = os.environ.get("OPENAI_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.openai_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        super().__init__(
            model=model, api_key=api_key, api_type=api_type, **kwargs
        )

        # Tokenizer
        import tiktoken
        # TODO: Remove.
        # Currently (17/09/2024) tiktoken doesn't support openai "o1" models.
        # https://github.com/openai/tiktoken/issues/337
        from tiktoken.model import MODEL_PREFIX_TO_ENCODING, MODEL_TO_ENCODING
        MODEL_PREFIX_TO_ENCODING.update({"o1-": "o200k_base"})
        if encoding_name is None:
            encoding_name = tiktoken.encoding_for_model(model).name
        self._encoding_name = encoding_name
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        # Special tokens
        self._allowed_special_tokens = allowed_special_tokens if allowed_special_tokens is not None \
            else self.default_allowed_special_tokens

        self.chat_mode = True  # Only OpenAI chat-mode will be supported

    def session(self, asynchronous=False):
        if asynchronous:
            return OpenAISession(self)
        else:
            return SyncSession(OpenAISession(self))

    @classmethod
    async def stream_then_save(cls, gen, key, stop_regex, n):
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
                if all_done:
                    gen.aclose()
                    break

        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out

    async def _library_call(self, **call_kwargs):
        """ Call the OpenAI API using the python package."""
        assert self.api_key is not None, "You must provide an OpenAI API key to use the OpenAI LLM. " \
                                         "Either pass it in the constructor, set the OPENAI_API_KEY environment " \
                                         "variable, or create the file ~/.openai_api_key with your key in it."

        # Filter non supported call arguments
        pass

        # Process messages
        messages = self.prompt_to_messages(call_kwargs['prompt'])
        call_kwargs['messages'] = messages
        assert 'prompt' in self._api_exclude_arguments

        # Parse call arguments
        call_args = self.parse_call_arguments(call_kwargs)

        # Start API client
        client = AsyncOpenAI(api_key=self.api_key)
        client._platform = PLATFORM

        # Call LLM API
        out = await client.chat.completions.create(**call_args)
        log.info(f"LLM call response: {out}")
        out = add_text_to_chat_mode(out)

        return out

    async def _rest_call(self, **kwargs):
        raise NotImplementedError

    async def _rest_stream_handler(self, response, session):
        raise NotImplementedError

    def encode(self, string: str, **kwargs) -> List[int]:
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self._allowed_special_tokens, **kwargs)

    def decode(self, tokens: List[int], **kwargs) -> str:
        return self._tokenizer.decode(tokens, **kwargs)


class OpenAISession(APILLMSession):
    pass


class _OpenAISession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1,
                       max_tokens=1000, max_completion_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the OpenAI API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the OpenAI API!"

        assert token_healing is None or token_healing is False, "The OpenAI API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
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

            functions = extract_function_defs(prompt)

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
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stop": stop,
                        "logprobs": logprobs,
                        "echo": echo,
                        "stream": stream,
                        **completion_kwargs
                    }

                    # "o1-":
                    #  - 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.
                    #  - 'temperature' does not support 0 with this model. Only the default (1) value is supported.
                    #  - 'stop' is not supported with this model.
                    #  - 'stream' does not support true with this model. Only the default (false) value is supported.
                    if call_args['model'].startswith('o1-'):
                        call_args.update({
                            "max_completion_tokens": max_completion_tokens,
                            "stream": False
                        })
                        del call_args['max_tokens']
                        del call_args['stop']

                    if functions is None:
                        if "function_call" in call_args:
                            del call_args["function_call"]
                    else:
                        call_args["functions"] = functions
                    if logit_bias is not None:
                        call_args["logit_bias"] = {str(k): v for k,v in logit_bias.items()} # convert keys to strings since that's the open ai api's format
                    out = await self.llm.caller(**call_args)

                    # "o1-":
                    # Response will be empty if couldn't complete the request within the 'max_completion_tokens'
                    # For now, we'll raise an error if this happens
                    if call_args['model'].startswith('o1-'):
                        if out['choices'][0].get('finish_reason', None) == 'length' \
                                and out['choices'][0].get('message', {}).get('content', None) == '':
                            raise Exception(f"Model 'o1-' returned empty response because couldn't "
                                            f"complete the request within 'max_completion_tokens': "
                                            f"{call_args['max_completion_tokens']}")

                except (openai.RateLimitError,
                        openai.APIConnectionError,
                        openai.APIStatusError,
                        openai.APIError,
                        openai.APITimeoutError) as err:
                    await asyncio.sleep(3)
                    error_msg = err.message
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) OpenAI API errors in a row! \n"
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


class OpenAIStreamer(BaseStreamer):
    def process_stream_chunk(self, chunk):
        raise NotImplementedError
