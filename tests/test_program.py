import copy
import json

from promptitude import guidance
from promptitude.guidance import ProgramState
import pytest
from .utils import get_llm


def test_chat_stream():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    async def f():
        chat = guidance("""<|im_start|>system
You are a helpful assistent.
<|im_end|>
<|im_start|>user
{{command}}
<|im_end|>
<|im_start|>assistant
{{gen 'answer' max_tokens=10 stream=True}}""")
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_chat_display():
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    async def f():
        chat = guidance("""<|im_start|>system
You are a helpful assistent.
<|im_end|>
<|im_start|>user
{{command}}
<|im_end|>
<|im_start|>assistant
{{gen 'answer' max_tokens=10}}""")
        out = await chat(command="How do I create a Fasttokenizer with hugging face auto?")
        assert len(out["answer"]) > 0
    loop.run_until_complete(f())

def test_agents():
    """Test agents, calling prompt twice"""

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    prompt = guidance('''<|im_start|>system
You are a helpful assistant.<|im_end|>
{{#geneach 'conversation' stop=False}}
<|im_start|>user
{{set 'this.user_text' (await 'user_text')}}<|im_end|>
<|im_start|>assistant
{{gen 'this.ai_text' n=1 temperature=0 max_tokens=900}}<|im_end|>{{/geneach}}''', echo=True)
    prompt = prompt(user_text='Hi there')
    assert len(prompt['conversation']) == 2
    prompt = prompt(user_text='Please help')
    assert len(prompt['conversation']) == 3

@pytest.mark.parametrize("llm", ["transformers:gpt2", ])
def test_stream_loop(llm):
    llm = get_llm(llm)
    program = guidance("""Generate a list of 5 company names:
{{#geneach 'companies' num_iterations=5~}}
{{@index}}. "{{gen 'this' max_tokens=5}}"
{{/geneach}}""", llm=llm)

    partials = []
    for p in program(stream=True, silent=True):
        partials.append(p.get("companies", []))
    assert len(partials) > 1
    assert len(partials[0]) < 5
    assert len(partials[-1]) == 5

@pytest.mark.parametrize("llm", ["transformers:gpt2", ])
def test_stream_loop_async(llm):
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    llm = get_llm(llm)

    async def f():
        program = guidance("""Generate a list of 5 company names:
{{#geneach 'companies' num_iterations=5~}}
{{@index}}. "{{gen 'this' max_tokens=5}}"
{{/geneach}}""", llm=llm)

        partials = []
        async for p in program(stream=True, async_mode=True, silent=True):
            partials.append(p.get("companies", []))
        assert len(partials) > 1
        assert len(partials[0]) < 5
        assert len(partials[-1]) == 5
    loop.run_until_complete(f())

@pytest.mark.parametrize("llm", ["openai:gpt-3.5-turbo", ])
def test_stream_loop(llm):
    llm = get_llm(llm)
    program = guidance("""
{{~#user~}}
Generate a list of 5 company names.
{{~/user~}}

{{#geneach 'companies' num_iterations=5~}}
{{#assistant~}}
{{gen 'this' max_tokens=5}}
{{~/assistant}}
{{/geneach}}""", llm=llm)

    partials = []
    for p in program(stream=True, silent=True):
        partials.append(p.get("companies", []))
    assert len(partials) > 1
    assert len(partials[0]) < 5
    assert len(partials[-1]) == 5

@pytest.mark.parametrize("llm", ["openai:gpt-3.5-turbo", ])
def test_stream_loop_async(llm):
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    import asyncio
    loop = asyncio.new_event_loop()

    llm = get_llm(llm)

    async def f():
        program = guidance("""
{{~#user~}}
Generate a list of 5 company names.
{{~/user~}}

{{#geneach 'companies' num_iterations=5~}}
{{#assistant~}}
{{gen 'this' max_tokens=5}}
{{~/assistant}}
{{/geneach}}""", llm=llm)

        partials = []
        async for p in program(stream=True, async_mode=True, silent=True):
            partials.append(p.get("companies", []))
        assert len(partials) > 1
        assert len(partials[0]) < 5
        assert len(partials[-1]) == 5
    loop.run_until_complete(f())

def test_logging_on():
    program = guidance("""This is a test prompt{{#if flag}} yes.{{/if}}""", log=True)
    executed_program = program(flag=True)
    assert len(executed_program.log) > 0

def test_logging_off():
    program = guidance("""This is a test prompt{{#if flag}} yes.{{/if}}""", log=False)
    executed_program = program(flag=True)
    assert executed_program.log is False


def test_async_mode_exceptions():
    """
    Ensures that exceptions in async_mode=True don't hang the program and are
    re-raised back to the caller.
    """
    import asyncio
    loop = asyncio.new_event_loop()

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    async def call_async():
        program = guidance("""
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
What is your name?
{{~/user}}

{{#assistant~}}
Hello my name is {{gen 'name' temperature=0 max_tokens=5}}.
{{~/assistant}}
""",
            async_mode=True
        )

        return await program()

    task = loop.create_task(call_async())
    completed_tasks, _ = loop.run_until_complete(
        asyncio.wait([task], timeout=5.0)
    )

    try:
        assert len(completed_tasks) == 1, "The task did not complete before timeout"
    finally:
        task.cancel()
        loop.run_until_complete(asyncio.sleep(0)) # give the loop a chance to cancel the tasks

    completed_task = list(completed_tasks)[0]

    assert isinstance(completed_task.exception(), AssertionError), \
        "Expect the exception to be propagated"

    loop.close()


def test_initial_state():
    """ Test that we can get the state of a Program and pass it as initial_state to a new Program """
    guidance.llm = get_llm('transformers:gpt2')

    # Create and execute a Program
    program = guidance("""Answer the following question: {{question}}
    {{gen 'answer' max_tokens=5}}""")
    executed_program = program(question="What is the capital of France?")

    # Get the state
    state = executed_program.state

    # Create a new Program with the initial_state
    new_program = guidance(initial_state=state)

    # Ensure that the new program has the same text and variables
    assert new_program.text == executed_program.text
    assert new_program.state == state


@pytest.mark.parametrize("llm", ["openai:gpt-4o-mini", ])
def test_initial_state_partial_execution(llm):
    """ Test that we can serialize a partially executed Program and continue execution with initial_state """
    llm = get_llm(llm)

    # Create a Program that uses 'await' to pause execution
    program = guidance("""
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}

    {{#user~}}
    Answer the following question: {{question}}
    {{~/user}}

    {{#assistant~}}
    {{#if response}}
    The answer is: {{response}}
    {{else}}
    Let me think about it.
    {{set 'response' (await 'response')}}
    The answer is: {{response}}
    {{/if}}
    {{~/assistant}}
    """, llm=llm)

    # Execute the Program without providing 'response', so it should pause and await
    partial_program = program(question="What is the capital of France?", await_missing=True)

    # Ensure that 'response' is missing
    assert 'response' not in partial_program.variables()
    assert "Let me think about it." in str(partial_program)
    assert "The answer is: Paris" not in str(partial_program)

    # Get the state
    state = partial_program.state

    # Now, suppose we receive the 'response' and want to continue execution
    # Create a new Program with initial_state, provide 'response', and execute to continue
    new_program = guidance(initial_state=state)

    # Continue the execution
    final_program = new_program(response="Paris")

    # Check that execution completed, and 'response' variable is set
    assert final_program['response'] == "Paris"
    assert "Let me think about it." not in str(final_program)
    assert "The answer is: Paris" in str(final_program)


@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:gpt-4o-mini", "anthropic:claude-3-haiku-20240307"])
def test_program_state(llm):
    llm = get_llm(llm)

    program = guidance(""" Program """, llm=llm)
    state = program.state
    state_dict = state.to_dict()
    assert isinstance(state_dict, dict)

    # Correct loading form dict
    alt_state_dict = copy.deepcopy(state_dict)
    alt_state = ProgramState.from_dict(alt_state_dict)
    assert state_dict == alt_state.to_dict()
    # Ensure json serialization
    json.dumps(state_dict)

    # Errors
    deleted_field = list(state_dict.keys())[0]
    del alt_state_dict[deleted_field]
    alt_state_dict.update({'non_existing': 'value'})
    with pytest.raises(ValueError) as exc_info:
        alt_state = ProgramState.from_dict(alt_state_dict)
    assert str(exc_info.value) == f"Missing fields: ['{deleted_field}']. Unused fields: ['non_existing']"
