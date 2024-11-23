def negate(x):
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
    >>> negate(5)
    -5
    >>> negate(-3)
    3
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"Unary '-' operator is not supported for type '{type(x).__name__}'.")

    return -x
