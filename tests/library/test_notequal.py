from promptitude import guidance


def test_notequal():
    """ Test the behavior of `notequal`.
    """
    program = guidance("""{{#if (notequal val 5)}}not equal{{else}}are equal{{/if}}""")
    assert str(program(val=4)) == "not equal"
    assert str(program(val=5)) == "are equal"
    assert str(program(val="5")) == "not equal"


def test_notequal_infix():
    """ Test the behavior of `notequal` using infix notation.
    """
    program = guidance("""{{#if val != 5}}not equal{{else}}are equal{{/if}}""")
    assert str(program(val=4)) == "not equal"
    assert str(program(val=5)) == "are equal"
    assert str(program(val="5")) == "not equal"
