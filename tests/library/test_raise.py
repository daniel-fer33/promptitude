from promptitude import guidance


def test_raise_with_msg():
    msg = "This is an error"
    program = guidance("{{raise msg}}")
    output = program(msg=msg)
    assert isinstance(output._exception, Exception)
    assert str(output._exception) == msg


def test_raise_with_formatted_msg():
    msg = "Error code: {code}"
    program = guidance("{{raise msg code=404}}")
    output = program(msg=msg, code=404)
    assert isinstance(output._exception, Exception)
    assert str(output._exception) == "Error code: 404"

    program = guidance("{{raise 'Error code: {code}' code=code}}")
    output = program(code=404)
    assert isinstance(output._exception, Exception)
    assert str(output._exception) == "Error code: 404"