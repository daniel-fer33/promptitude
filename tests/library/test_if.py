from promptitude import guidance


def test_if():
    """ Test the behavior of `if`.
    """

    prompt = guidance("""Answer: {{#if flag}}Yes{{/if}}""")

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: Yes"

    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: "


def test_if_complex_block():
    prompt = guidance("""Answer: {{#if True}}Yes {{my_var}} we{{/if}}""")

    out = prompt(my_var="then")

    assert str(out) == "Answer: Yes then we"


def test_if_else():
    """ Test the behavior of `if` with an `else` clause.
    """

    prompt = guidance("""Answer 'Yes' or 'No': '{{#if flag}}Yes{{else}}No{{/if}}'""")

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'Yes'"

    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'No'"


def test_if_complex_blockwith_else():
    prompt = guidance("""Answer: {{#if flag}}Yes {{my_var}} we{{else}}No {{my_var}}{{/if}}""")

    out = prompt(my_var="then", flag=True)
    assert str(out) == "Answer: Yes then we"

    out = prompt(my_var="then", flag=False)
    assert str(out) == "Answer: No then"


def test_elif_else():
    """ Test the behavior of `if` with an `else` clause.
    """
    prompt = """Answer 'Yes' or 'No': '{{#if flag}}Yes{{elif flag2}}maybe{{else}}No{{/if}}'"""

    program = guidance(prompt)
    out = program(flag=True, flag2=True)
    assert str(out) == "Answer 'Yes' or 'No': 'Yes'"

    program = guidance(prompt)
    out = program(flag=True, flag2=False)
    assert str(out) == "Answer 'Yes' or 'No': 'Yes'"

    program = guidance(prompt)
    out = program(flag=False, flag2=True)
    assert str(out) == "Answer 'Yes' or 'No': 'maybe'"

    program = guidance(prompt)
    out = program(flag=False, flag2=False)
    assert str(out) == "Answer 'Yes' or 'No': 'No'"

    prompt = """Answer 'Yes' or 'No': '{{#if flag}}Yes{{elif flag2}}maybe{{elif flag3}}No way!{{else}}I dont' know{{/if}}'"""

    program = guidance(prompt)
    out = program(flag=False, flag2=False, flag3=True)
    assert str(out) == "Answer 'Yes' or 'No': 'No way!'"

    program = guidance(prompt)
    out = program(flag=False, flag2=False, flag3=False)
    assert str(out) == "Answer 'Yes' or 'No': 'I dont' know'"
