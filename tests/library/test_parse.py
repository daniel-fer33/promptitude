import guidance


def test_parse():
    """ Test the basic behavior of `parse`.
    """

    program = guidance("""This is parsed: {{parse template}}""")
    assert str(program(template="My name is {{name}}", name="Bob")) == "This is parsed: My name is Bob"


def test_parse_with_name():
    program = guidance("""This is parsed: {{parse template name="parsed"}}""")
    executed_program = program(template="My name is {{name}}", name="Bob")
    assert executed_program["parsed"] == "My name is Bob"


def test_parse_with_hidden_prefix():
    llm = guidance.llms.Mock("Marley")
    program = guidance(
        """This is executed: {{parse template name="exec_out" hidden_prefix=True}}. Repeated: {{gen 'repeated' save_prompt='saved_prompt_2'}}""",
        llm=llm)
    out = program(template="My name is {{name}} {{gen 'surname' save_prompt='saved_prompt_1'}}", name="Bob")
    assert str(out) == "This is executed: My name is Bob Marley. Repeated: Marley"

    variables = out.variables()
    assert variables['exec_out'] == 'My name is Bob Marley'
    assert guidance(variables['saved_prompt_1']).text == 'My name is Bob '
    assert guidance(variables['saved_prompt_2']).text == 'This is executed: My name is Bob Marley. Repeated: '


def test_parse_with_hidden_prefix_and_hidden_execution():
    llm = guidance.llms.Mock("Marley")
    program = guidance(
        """This is executed: {{parse template name="exec_out" hidden_prefix=True hidden=True}}. Repeated: {{gen 'repeated' save_prompt='saved_prompt_2'}}""",
        llm=llm)
    out = program(template="My name is {{name}} {{gen 'surname' save_prompt='saved_prompt_1'}}", name="Bob")
    assert str(out) == "This is executed: . Repeated: Marley"

    variables = out.variables()
    assert variables['exec_out'] == 'My name is Bob Marley'
    assert guidance(variables['saved_prompt_1']).text == 'My name is Bob '
    assert guidance(variables['saved_prompt_2']).text == 'This is executed: . Repeated: '
