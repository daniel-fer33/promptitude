from typing import Any


def not_(value: Any) -> bool:
    """
    Negate the given value.

    Parameters
    ----------
    value : Any
        The value to be negated.

    Returns
    -------
    result : bool
        The negation of the given value, as a boolean.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Negation: {{set 'result' (not_ value)}}")
    >>> output = program(value=True)
    >>> print(output["result"])
    False
    """
    return not value
