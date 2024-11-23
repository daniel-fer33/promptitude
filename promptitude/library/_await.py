from typing import Any, Dict, Union


async def await_(name: str, consume: bool = True, _parser_context: Union[Dict, None] = None) -> Any:
    """
    Awaits a variable by returning its value and then optionally deleting it.

    This function is useful for repeatedly getting values since programs
    will pause when they need a value that is not yet set. Placing `await_`
    in a loop creates a stateful "agent" that can repeatedly await values
    when called multiple times.

    Parameters
    ----------
    name : str
        The name of the variable to await.
    consume : bool, optional
        Whether to delete the variable from the variable stack after returning its value.
        Defaults to True.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    Any
        The value of the awaited variable.

    Raises
    ------
    TypeError
        If 'name' is not a string.
        If 'consume' is not a boolean.
    ValueError
        If '_parser_context' is None.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> prompt = guidance("User response: '{{await_('user_response')}}'")
    >>> # The program will wait until 'user_response' is provided
    >>> output = prompt(user_response='Yes')
    >>> print(output)
    User response: 'Yes'
    """
    if not isinstance(name, str):
        raise TypeError(f"'name' must be of type str, got {type(name)}.")
    if not isinstance(consume, bool):
        raise TypeError(f"'consume' must be of type bool, got {type(consume)}.")
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")

    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']
    if name not in variable_stack:
        parser.executing = False
    else:
        value = variable_stack[name]
        if consume:
            del variable_stack[name]
        return value
