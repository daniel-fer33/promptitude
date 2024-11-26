from promptitude import guidance


def test_not():
    """ Basic test of `not`.
    """

    program = guidance("""Negation: {{set 'result' (not value)}}""")
    assert program(value=True)["result"] is False
    assert program(value=False)["result"] is True
    assert program(value=0)["result"] is True
    assert program(value=1)["result"] is False
    assert program(value=None)["result"] is True
    assert program(value="")["result"] is True
    assert program(value="non-empty string")["result"] is False
    assert program(value=[])[
        "result"] is True  # Empty list should be negated to True
    assert program(value=[1, 2, 3])["result"] is False  # Non-empty list
