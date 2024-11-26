from numbers import Number


def add(*args: Number) -> Number:
    """
    Add the given variables together.

    Parameters
    ----------
    *args : Number
        The numerical values to be added together.

    Returns
    -------
    result : Number
        The sum of the given values.

    Raises
    ------
    TypeError
        If any of the arguments are not numbers.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Result: {{set 'result' (add 20 variable)}}")
    >>> output = program(variable=10)
    >>> print(output["result"])
    30
    """
    for arg in args:
        if not isinstance(arg, Number):
            raise TypeError(f"All arguments must be numbers, got {type(arg)}")
    return sum(args)
