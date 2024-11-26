from promptitude import guidance

def test_range():
    """ Basic tests of `range`.
    """

    # Test range with a single argument
    program = guidance("Numbers: {{range(5)}}")
    output = program()
    assert str(output) == "Numbers: [0, 1, 2, 3, 4]"

    # Test range with start and stop arguments
    program = guidance("Numbers: {{range(1, 6)}}")
    output = program()
    assert str(output) == "Numbers: [1, 2, 3, 4, 5]"

    # Test range with start, stop, and step arguments
    program = guidance("Numbers: {{range(0, 10, 2)}}")
    output = program()
    assert str(output) == "Numbers: [0, 2, 4, 6, 8]"

    # Test range with negative step
    program = guidance("Numbers: {{range(5, 0, -1)}}")
    output = program()
    assert str(output) == "Numbers: [5, 4, 3, 2, 1]"

    # Test range with variables
    program = guidance("Numbers: {{range(n)}}")
    output = program(n=3)
    assert str(output) == "Numbers: [0, 1, 2]"
