from typing import Any


def equal(*args: Any) -> bool:
    """
    Check that all arguments are equal.

    Parameters
    ----------
    *args : Any
        The values to compare.

    Returns
    -------
    result : bool
        True if all arguments are equal, False otherwise.

    Raises
    ------
    ValueError
        If less than two arguments are provided.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance('{{#if (equal val 5)}}are equal{{else}}not equal{{/if}}')
    >>> output = program(val=4)
    >>> print(output)
    not equal
    >>> output = program(val=5)
    >>> print(output)
    are equal
    """
    if len(args) < 2:
        raise ValueError("equal() requires at least two arguments.")

    first_arg = args[0]
    for arg in args[1:]:
        if arg != first_arg:
            return False
    return True
