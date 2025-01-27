from promptitude import guidance


def test_contains():
    """ Test the behavior of `contains`.
    """

    program = guidance("""{{#if (contains val "substr")}}are equal{{else}}not equal{{/if}}""")
    assert str(program(val="no sub")) == "not equal"
    assert str(program(val="this is a substr")) == "are equal"
    assert str(program(val="this is a subsr")) == "not equal"

    program_list = guidance("""{{#if (contains val 2)}}list contains 2{{else}}no 2 in list{{/if}}""")
    assert str(program_list(val=[1, 2, 3])) == "list contains 2"
    assert str(program_list(val=[1, 3])) == "no 2 in list"

    # Test with dictionary (checking keys)
    program_dict = guidance(
        """{{#if (contains val "test_key")}}dict contains "test_key"{{else}}no "test_key" in dict{{/if}}""")
    assert str(program_dict(val={"test_key": 123})) == 'dict contains "test_key"'
    assert str(program_dict(val={"other_key": 456})) == 'no "test_key" in dict'
