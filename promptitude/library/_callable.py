from typing import Any, Optional, Dict

import builtins


def callable(value: Any, _parser_context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if the given value is callable.

    Parameters
    ----------
    value : Any
        The value to check if it is callable.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    is_callable : bool
        True if the value is callable, False otherwise.

    Raises
    ------
    ValueError
        If '_parser_context' is None.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Is callable: {{callable some_function}}")
    >>> output = program(some_function=lambda x: x + 1)
    >>> print(output)
    True
    """
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")

    variable_stack = _parser_context["variable_stack"]

    function_call = variable_stack["llm.extract_function_call"](value)
    if function_call is not None:
        return True

    return builtins.callable(value)
