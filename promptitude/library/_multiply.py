from numbers import Number
import math


def multiply(*args: Number) -> Number:
    """
    Multiply the given variables together.

    Parameters
    ----------
    *args : Number
        The numerical values to be multiplied together.

    Returns
    -------
    result : Number
        The product of the given values.

    Raises
    ------
    TypeError
        If any of the arguments are not numbers.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Result: {{set 'result' (multiply 5 variable)}}")
    >>> output = program(variable=4)
    >>> print(output["result"])
    20
    """
    for arg in args:
        if not isinstance(arg, Number):
            raise TypeError(f"All arguments must be numbers, got {type(arg)}")
    return math.prod(args)
