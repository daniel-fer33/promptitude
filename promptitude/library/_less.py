from numbers import Number


def less(arg1: Number, arg2: Number) -> bool:
    """
    Check if `arg1` is less than `arg2`.

    Parameters
    ----------
    arg1 : Number
        The first numerical value to compare.
    arg2 : Number
        The second numerical value to compare.

    Returns
    -------
    result : bool
        `True` if `arg1` is less than `arg2`, `False` otherwise.

    Raises
    ------
    TypeError
        If either of the arguments is not a number.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("{{#if (less val 5)}}less{{else}}not less{{/if}}")
    >>> print(program(val=6))
    not less
    >>> print(program(val=4))
    less

    Note that this function can also be called using `<` as well as `less`.
    """
    if not isinstance(arg1, Number):
        raise TypeError(f"'arg1' must be a number, got {type(arg1)}.")
    if not isinstance(arg2, Number):
        raise TypeError(f"'arg2' must be a number, got {type(arg2)}.")
    return arg1 < arg2
