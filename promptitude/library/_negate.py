from numbers import Number


def negate(x: Number) -> Number:
    """
    Returns the negation of the given numeric value.

    Parameters
    ----------
    x : int or float
        The number to negate.

    Returns
    -------
    int or float
        The negated value.

    Raises
    ------
    TypeError
        If the argument is not a number.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Result: {{set 'result' (-variable)}}")
    >>> output = program(variable=5)
    >>> print(output["result"])
    -5
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"Unary '-' operator is not supported for type '{type(x).__name__}'.")

    return -x
