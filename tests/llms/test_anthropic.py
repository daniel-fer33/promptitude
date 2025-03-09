import unittest
import importlib

from promptitude import guidance
from promptitude.llms import Anthropic
from ..utils import get_llm


def test_geneach_claude_3():
    """ Test a geneach loop with Claude 3.
    """

    guidance.llm = get_llm("anthropic:claude-3-haiku-20240307")

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
    print(out)
    assert len(out["conversation"]) == 2


def test_syntax_match():
    """ Test a syntax.
    """

    guidance.llm = get_llm("anthropic:claude-3-haiku-20240307")

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


def test_nostream():
    guidance.llm =get_llm("anthropic:claude-3-haiku-20240307")
    prompt = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=False}}
{{~/assistant}}''', stream=False)
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0


def test_stream():
    guidance.llm =get_llm("anthropic:claude-3-haiku-20240307")
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


class TestAnthropicSerialization(unittest.TestCase):
    def test_serialize(self):
        # Create an instance of Anthropic with some parameters
        anthropic_instance = Anthropic(
            model='claude-v1',
            api_key='test_api_key',
            temperature=0.5,
            rest_call=True,
            api_base='https://api.anthropic.com/v1',
            api_version='2023-01-01',
        )

        # Serialize the instance
        serialized = anthropic_instance.serialize()

        # Check that the serialized output is a dictionary
        self.assertIsInstance(serialized, dict)

        # Ensure 'module_name', 'class_name', and 'init_args' are present
        self.assertIn('module_name', serialized)
        self.assertIn('class_name', serialized)
        self.assertIn('init_args', serialized)

        # Verify that 'module_name' and 'class_name' have correct values
        self.assertEqual(serialized['module_name'], anthropic_instance.__module__)
        self.assertEqual(serialized['class_name'], anthropic_instance.__class__.__name__)

        # Ensure 'init_args' is a dictionary
        self.assertIsInstance(serialized['init_args'], dict)

        # Ensure 'api_key' and 'token' are excluded from 'init_args'
        self.assertNotIn('api_key', serialized['init_args'])
        self.assertNotIn('token', serialized['init_args'])

        # Ensure that class_attribute_map is correctly applied in 'init_args'
        for arg, attr_name in anthropic_instance.class_attribute_map.items():
            self.assertIn(arg, serialized['init_args'])
            self.assertEqual(serialized['init_args'][arg], getattr(anthropic_instance, attr_name))

        # Create a new instance using the 'init_args'
        module_name, class_name = serialized['module_name'], serialized['class_name']
        cls = getattr(importlib.import_module(module_name), class_name)
        new_anthropic_instance = cls(**serialized['init_args'])

        # Check that the new instance has the same attributes as the original
        self.assertEqual(new_anthropic_instance.model_name, anthropic_instance.model_name)
        self.assertEqual(new_anthropic_instance.temperature, anthropic_instance.temperature)
        self.assertEqual(new_anthropic_instance.rest_call, anthropic_instance.rest_call)
        self.assertEqual(new_anthropic_instance.api_type, anthropic_instance.api_type)
        self.assertEqual(new_anthropic_instance.api_base, anthropic_instance.api_base)
        self.assertEqual(new_anthropic_instance.api_version, anthropic_instance.api_version)


class TestThinkingModels:
    def test_claude_37_thinking(self):
        llm = get_llm("anthropic:claude-3-7-sonnet-20250219")  # Test will be skipped if key not found

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
        {{gen 'expert_names' temperature=0 max_tokens=300}}
        {{~/assistant}}

        {{#user~}}
        Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' llm_alt_model="claude-3-7-sonnet-20250219" temperature=1 max_tokens=2048 thinking={"type": "enabled", "budget_tokens": 1024} }}
        {{~/assistant}}
        '''

        query = 'How can I be more productive?'

        program = guidance(guide, llm=llm, caching=False, silent=True, stream=False, log=True)

        out = program(query=query)
        assert not out._exception
