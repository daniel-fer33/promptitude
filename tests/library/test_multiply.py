from promptitude import guidance


def test_multiply():
    """ Basic test of `multiply`.
    """

    program = guidance("""Write a number: {{set 'user_response' (multiply 5 variable)}}""")
    assert program(variable=4)["user_response"] == 20
    assert program(variable=2.5)["user_response"] == 12.5


def test_multiply_multi():
    """ Test more than 2 arguments for `multiply`.
    """

    program = guidance("""Write a number: {{set 'user_response' (multiply 5 2 variable)}}""")
    assert program(variable=4)["user_response"] == 40
    assert program(variable=2.5)["user_response"] == 25.0


def test_multiply_infix():
    """ Basic infix test of `multiply`.
    """

    program = guidance("""Write a number: {{set 'user_response' 5 * variable}}""")
    assert program(variable=4)["user_response"] == 20
    assert program(variable=2.5)["user_response"] == 12.5
