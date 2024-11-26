from typing import List

import builtins


def range(*args: int, **kwargs: int) -> List:
    """
    Build a range of numbers.

    Parameters
    ----------
    *args : int
        Positional arguments to define the range (start, stop, step).
    **kwargs : int
        Keyword arguments to define the range ('start', 'stop', 'step').

    Returns
    -------
    range_object : range
        A range object representing the sequence of integers.

    Raises
    ------
    TypeError
        If any of the arguments are not integers.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Numbers: {{list(range(5))}}")
    >>> output = program()
    >>> print(output)
    Numbers: [0, 1, 2, 3, 4]
    """
    for arg in args:
        if not isinstance(arg, int):
            raise TypeError(f"All positional arguments must be integers, got {type(arg)}.")
    for key, value in kwargs.items():
        if not isinstance(value, int):
            raise TypeError(f"All keyword arguments must be integers, got {type(value)} for '{key}'.")
    return list(builtins.range(*args, **kwargs))
