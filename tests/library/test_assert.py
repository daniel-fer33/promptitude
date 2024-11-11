import guidance


def test_assert_true():
    """ Test the behavior of `equal`.
    """

    # True, None
    program = guidance("""{{assert True}}""")
    output = program()
    assert output._exception == None


def test_assert_false():
    # False, None
    program = guidance("""{{assert False}}""")
    output = program()
    assert isinstance(output._exception, AssertionError)


def test_assert_true_msg():
    # True, msg
    msg = "This is not an error"
    program = guidance("""{{assert True msg}}""")
    output = program(msg=msg)
    assert output._exception == None


def test_assert_false_msg():
    # False, None
    msg = "This is an error"
    program = guidance("""{{assert False msg}}""")
    output = program(msg=msg)
    assert isinstance(output._exception, AssertionError)
    err = output._exception
    assert str(output._exception) == msg
