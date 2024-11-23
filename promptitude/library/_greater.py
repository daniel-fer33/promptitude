from numbers import Number


def greater(arg1: Number, arg2: Number) -> bool:
    """
    Check if arg1 is greater than arg2.

    Note that this can also be called using `>` as well as `greater`.

    Parameters
    ----------
    arg1 : Number
        The first number to compare.
    arg2 : Number
        The second number to compare.

    Returns
    -------
    result : bool
        True if arg1 is greater than arg2, False otherwise.

    Raises
    ------
    TypeError
        If arg1 or arg2 are not numbers.

    Examples
    --------
    >>> from promptitude import guidance
    >>> program = guidance("{{#if (greater val 5)}}greater{{else}}not greater{{/if}}")
    >>> output = program(val=6)
    >>> print(output)
    greater
    """
    if not isinstance(arg1, Number):
        raise TypeError(f"arg1 must be a number, got {type(arg1)}")
    if not isinstance(arg2, Number):
        raise TypeError(f"arg2 must be a number, got {type(arg2)}")
    return arg1 > arg2
