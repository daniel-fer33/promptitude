from numbers import Real


def subtract(minuend: Real, subtrahend: Real) -> Real:
    """
    Subtract the second variable from the first.

    Parameters
    ----------
    minuend : Real
        The number to subtract from.
    subtrahend : Real
        The number to subtract.

    Returns
    -------
    result : Real
        The result of the subtraction.

    Raises
    ------
    TypeError
        If `minuend` or `subtrahend` are not real numbers.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Result: {{set 'result' (subtract 20 variable)}}")
    >>> output = program(variable=10)
    >>> print(output["result"])
    10
    """
    if not isinstance(minuend, Real):
        raise TypeError(f"'minuend' must be a real number, got {type(minuend).__name__}")
    if not isinstance(subtrahend, Real):
        raise TypeError(f"'subtrahend' must be a real number, got {type(subtrahend).__name__}")
    return minuend - subtrahend
