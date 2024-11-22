import unittest

from promptitude import guidance
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
        anthropic_instance = guidance.llms.Anthropic(
            model='claude-v1',
            api_key='test_api_key',
            temperature=0.5,
            rest_call=True,
            api_type='anthropic_test',
            api_base='https://api.anthropic.com/v1',
            api_version='2023-01-01',
            allowed_special_tokens={'<|endoftext|>', '<|custom_token|>'}
        )

        # Serialize the instance
        serialized = anthropic_instance.serialize()

        # Check that the serialized output is a dictionary
        self.assertIsInstance(serialized, dict)

        # Ensure 'api_key' and 'token' are excluded
        self.assertNotIn('api_key', serialized)
        self.assertNotIn('token', serialized)

        # Ensure that class_attribute_map is correctly applied
        for arg, attr_name in anthropic_instance.class_attribute_map.items():
            self.assertIn(arg, serialized)
            self.assertEqual(serialized[arg], getattr(anthropic_instance, attr_name))

        # Create a new instance using the serialized dict
        new_anthropic_instance = guidance.llms.Anthropic(**serialized)

        # Check that the new instance has the same attributes as the original
        self.assertEqual(new_anthropic_instance.model_name, anthropic_instance.model_name)
        self.assertEqual(new_anthropic_instance.temperature, anthropic_instance.temperature)
        self.assertEqual(new_anthropic_instance.rest_call, anthropic_instance.rest_call)
        self.assertEqual(new_anthropic_instance.api_type, anthropic_instance.api_type)
        self.assertEqual(new_anthropic_instance.api_base, anthropic_instance.api_base)
        self.assertEqual(new_anthropic_instance.api_version, anthropic_instance.api_version)
        self.assertEqual(new_anthropic_instance.allowed_special_tokens, anthropic_instance.allowed_special_tokens)
