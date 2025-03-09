import unittest
import re
import importlib
import pytest

from promptitude import guidance
from promptitude.llms import OpenAI
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
        'o1-mini-2024-09-12',
        'o1',
        'chatgpt-4o-latest'
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

@pytest.mark.parametrize("llm_name", [
    "openai:gpt-3.5-turbo",
    "openai:gpt-4o-mini",
    #"openai:o1",
    "openai:chatgpt-4o-latest",
])
def test_models(llm_name):
    guidance.llm = get_llm(llm_name)

    chat_loop = guidance('''
    {{#system~}}
    You are a helpful assistant
    {{~/system}}

    {{#user~}}
    Hello!
    {{~/user}}

    {{#assistant~}}
    {{gen 'response' temperature=0 max_tokens=3}}
    {{~/assistant}}
    
    {{#user~}}
    Hello again!
    {{~/user}}

    {{#assistant~}}
    {{gen 'response' temperature=0 max_tokens=3 llm_alt_model=alt_model }}
    {{~/assistant}}
    ''')

    out = chat_loop(alt_model=llm_name.split(':')[1])
    assert 'response' in out.variables()


class TestOpenAISerialization(unittest.TestCase):
    def test_serialize(self):
        # Create an instance of OpenAI with some parameters
        openai_instance = OpenAI(
            model='gpt-3.5-turbo',
            api_key='test_api_key',
            api_base='https://api.openai.com/v1',
            temperature=0.7,
            organization='test_org',
            rest_call=True,
            allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>", "<|custom_token|>"},
            encoding_name='p50k_base'
        )

        # Serialize the instance
        serialized = openai_instance.serialize()

        # Check that the serialized output is a dictionary
        self.assertIsInstance(serialized, dict)

        # Ensure 'module_name', 'class_name', and 'init_args' are present
        self.assertIn('module_name', serialized)
        self.assertIn('class_name', serialized)
        self.assertIn('init_args', serialized)

        # Verify that 'module_name' and 'class_name' have correct values
        self.assertEqual(serialized['module_name'], openai_instance.__module__)
        self.assertEqual(serialized['class_name'], openai_instance.__class__.__name__)

        # Ensure 'init_args' is a dictionary
        self.assertIsInstance(serialized['init_args'], dict)

        # Ensure 'api_key' and 'token' are excluded from 'init_args'
        self.assertNotIn('api_key', serialized['init_args'])
        self.assertNotIn('token', serialized['init_args'])

        # Ensure that class_attribute_map is correctly applied in 'init_args'
        for arg, attr_name in openai_instance.class_attribute_map.items():
            self.assertIn(arg, serialized['init_args'])
            self.assertEqual(serialized['init_args'][arg], getattr(openai_instance, attr_name))

        # Create a new instance using the 'init_args'
        module_name, class_name = serialized['module_name'], serialized['class_name']
        cls = getattr(importlib.import_module(module_name), class_name)
        new_openai_instance = cls(**serialized['init_args'])

        # Check that the new instance has the same attributes as the original
        self.assertEqual(new_openai_instance.model_name, openai_instance.model_name)
        self.assertEqual(new_openai_instance.api_base, openai_instance.api_base)
        self.assertEqual(new_openai_instance.temperature, openai_instance.temperature)
        self.assertEqual(new_openai_instance.chat_mode, openai_instance.chat_mode)
        self.assertEqual(new_openai_instance.organization, openai_instance.organization)
        self.assertEqual(new_openai_instance.rest_call, openai_instance.rest_call)
        self.assertEqual(new_openai_instance._allowed_special_tokens, openai_instance._allowed_special_tokens)
        self.assertEqual(new_openai_instance._encoding_name, openai_instance._encoding_name)


class TestThinkingModels:
    def test_o3_mini_thinking(self):
        guide = '''
        {{#system~}}
        You are a helpful and terse assistant.
        {{~/system}}

        {{#user~}}
        I want a response to the following question:
        {{query}}
        Name 3 world-class experts (past or present) who would be great at answering this?
        Don't answer the question yet.
        {{~/user}}

        {{#assistant~}}
        {{gen 'expert_names' max_completion_tokens=1000}}
        {{~/assistant}}

        {{#user~}}
        Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' max_completion_tokens=5000 reasoning_effort="low"}}
        {{~/assistant}}
        '''

        query = 'How can I be more productive?'

        llm = guidance.llms.OpenAI(
            "o3-mini"
        )

        program = guidance(guide, llm=llm, caching=False, silent=True, stream=False, log=True)

        out = program(query=query)
        assert not out._exception
