import unittest
import re

from promptitude import guidance
from ..utils import get_llm


def test_chat_model_pattern():
    chat_models = [
        'gpt-4o-mini',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o',
        'gpt-4o-2024-05-13',
        'gpt-4-turbo',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-turbo-preview',
        'gpt-4-0125-preview',
        'gpt-4-vision-preview',
        'gpt-4-1106-vision-preview',
        'gpt-4',
        'gpt-4-0613',
        'gpt-4-32k',
        'gpt-4-32k-0613',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-instruct',
        'gpt-3.5-turbo-16k-0613',
        'o1-preview',
        'o1-preview-2024-09-12',
        'o1-mini',
        'o1-mini-2024-09-12'
    ]

    chat_model_pattern = guidance.llms.OpenAI.chat_model_pattern
    all_good = True
    failed = []
    for model in chat_models:
        if re.match(chat_model_pattern, model):
            chat_mode = True
        else:
            chat_mode = False
            failed.append(model)

        all_good = all_good and chat_mode

    assert all_good, f"Model(s) not recognized: {failed}"


def test_geneach_chat_gpt():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    chat_loop = guidance('''
{{#system~}}
You are a helpful assistant
{{~/system}}

{{~#geneach 'conversation' stop=False}}
{{#user~}}
This is great!
{{~/user}}

{{#assistant~}}
{{gen 'this.response' temperature=0 max_tokens=3}}
{{~/assistant}}
{{#if @index > 0}}{{break}}{{/if}}
{{~/geneach}}''')

    out = chat_loop()
    assert len(out["conversation"]) == 2


def test_syntax_match():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    chat_loop = guidance('''
{{~#system~}}
You are a helpful assistant
{{~/system~}}

{{~#user~}}
This is great!
{{~/user~}}

{{~#assistant~}}
Indeed
{{~/assistant~}}''')

    out = chat_loop()
    assert str(out) == '<|im_start|>system\nYou are a helpful assistant<|im_end|><|im_start|>user\nThis is great!<|im_end|><|im_start|>assistant\nIndeed<|im_end|>'


def test_rest_nostream():
    guidance.llm = get_llm('openai:babbage-002', endpoint="https://api.openai.com/v1/completions", rest_call=True)
    a = guidance('''Hello,  my name is{{gen 'name' stream=False max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0


def test_rest_stream():
    guidance.llm = get_llm('openai:babbage-002', endpoint="https://api.openai.com/v1/completions", rest_call=True)
    a = guidance('''Hello,  my name is{{gen 'name' stream=True max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0


def test_rest_chat_nostream():
    guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
    prompt = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=False}}
{{~/assistant}}''')
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0


def test_rest_chat_stream():
    guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
    prompt = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=True}}
{{~/assistant}}''')
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0


class TestOpenAISerialization(unittest.TestCase):

    def test_serialize(self):
        # Create an instance of OpenAI with some parameters
        openai_instance = guidance.llms.OpenAI(
            model='gpt-3.5-turbo',
            api_key='test_api_key',
            api_base='https://api.openai.com/v1',
            temperature=0.7,
            chat_mode=True,
            organization='test_org',
            rest_call=True,
            allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>", "<|custom_token|>"},
            endpoint='https://api.openai.com/v1',
            encoding_name='p50k_base'
        )

        # Serialize the instance
        serialized = openai_instance.serialize()

        # Check that the serialized output is a dictionary
        self.assertIsInstance(serialized, dict)

        # Ensure 'api_key' and 'token' are excluded
        self.assertNotIn('api_key', serialized)
        self.assertNotIn('token', serialized)

        # Ensure that class_attribute_map is correctly applied
        for arg, attr_name in openai_instance.class_attribute_map.items():
            self.assertIn(arg, serialized)
            self.assertEqual(serialized[arg], getattr(openai_instance, attr_name))

        # Create a new instance using the serialized dict
        new_openai_instance = guidance.llms.OpenAI(**serialized)

        # Check that the new instance has the same attributes as the original
        self.assertEqual(new_openai_instance.model_name, openai_instance.model_name)
        self.assertEqual(new_openai_instance.api_base, openai_instance.api_base)
        self.assertEqual(new_openai_instance.temperature, openai_instance.temperature)
        self.assertEqual(new_openai_instance.chat_mode, openai_instance.chat_mode)
        self.assertEqual(new_openai_instance.organization, openai_instance.organization)
        self.assertEqual(new_openai_instance.rest_call, openai_instance.rest_call)
        self.assertEqual(new_openai_instance.allowed_special_tokens, openai_instance.allowed_special_tokens)
        self.assertEqual(new_openai_instance.endpoint, openai_instance.endpoint)
        self.assertEqual(new_openai_instance._encoding_name, openai_instance._encoding_name)
