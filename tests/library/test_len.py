from promptitude import guidance


def test_len_list():
    """ Test `len` with a list.
    """
    program = guidance("""Length: {{len variable}}""")
    out = program(variable=[1, 2, 3, 4])
    assert str(out) == "Length: 4"


def test_len_string():
    """ Test `len` with a string.
    """
    program = guidance("""Length: {{len variable}}""")
    out = program(variable="hello")
    assert str(out) == "Length: 5"


def test_len_dict():
    """ Test `len` with a dictionary.
    """
    program = guidance("""Length: {{len variable}}""")
    out = program(variable={"a": 1, "b": 2, "c": 3})
    assert str(out) == "Length: 3"


def test_len_empty():
    """ Test `len` with an empty list.
    """
    program = guidance("""Length: {{len variable}}""")
    out = program(variable=[])
    assert str(out) == "Length: 0"


def test_len_no_len():
    """ Test `len` with an object that does not support `len()`.
    """
    program = guidance("""Length: {{len variable}}""")
    output = program(variable=42)
    assert isinstance(output._exception, TypeError)
