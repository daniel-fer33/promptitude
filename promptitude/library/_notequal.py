from typing import Any


def notequal(arg1: Any, arg2: Any) -> bool:
    """
    Check that the arguments are not equal.

    Parameters
    ----------
    arg1 : Any
        The first argument to compare.
    arg2 : Any
        The second argument to compare.

    Returns
    -------
    result : bool
        True if the arguments are not equal, False otherwise.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Result: {{notequal 5 variable}}")
    >>> output = program(variable=10)
    >>> print(output["result"])
    True

    >>> output = program(variable=5)
    >>> print(output["result"])
    False
    """
    return arg1 != arg2
