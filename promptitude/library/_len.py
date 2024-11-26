from typing import Any

import builtins


def len(s: Any) -> int:
    """
    Return the length of the given object.

    Parameters
    ----------
    s : Any
        The object whose length is to be returned. It should be an object that supports the `len()` function.

    Returns
    -------
    length : int
        The length of the object.

    Raises
    ------
    TypeError
        If the object does not support the `len()` function.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Length: {{len variable}}")
    >>> output = program(variable=[1, 2, 3])
    >>> print(output)
    Length: 3
    """
    try:
        return builtins.len(s)
    except TypeError as e:
        raise TypeError(f"Object of type '{type(s).__name__}' has no len()") from e
