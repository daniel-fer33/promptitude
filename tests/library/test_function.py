from promptitude import guidance


def test_function():
    """ Basic test of `function`.
    """
    llm = guidance.llms.Mock("function output")
    program = guidance("""
{{~#system}}System prompt here.{{/system}}
{{~#function}}Function content here.{{/function}}
{{~#assistant}}{{gen 'output' save_prompt='prompt'}}{{/assistant}}""", llm=llm)
    out = program()
    assert '<|im_start|>function\nFunction content here.<|im_end|>' in str(out)
    assert out["output"] == "function output"
