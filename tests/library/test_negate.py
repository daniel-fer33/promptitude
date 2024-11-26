from promptitude import guidance


def test_negate_positive():
    """Test negation of a positive number."""
    program = guidance("{{set 'result' (-value)}}")
    output = program(value=5)
    assert output["result"] == -5


def test_negate_negative():
    """Test negation of a negative number."""
    program = guidance("{{set 'result' (-value)}}")
    output = program(value=-10)
    assert output["result"] == 10


def test_negate_zero():
    """Test negation of zero."""
    program = guidance("{{set 'result' (-value)}}")
    output = program(value=0)
    assert output["result"] == 0


def test_negate_non_numeric():
    """Test negation of a non-numeric input."""
    program = guidance("{{set 'result' (-value)}}")
    output = program(value='abc')
    assert isinstance(output._exception, TypeError)
