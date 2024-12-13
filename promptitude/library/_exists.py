from typing import Any, Dict, Union
from promptitude._variable_stack import VariableStack


def exists(name: str, check_nested: bool = False, _parser_context: Union[Dict, None] = None) -> bool:
    """
    Return whether a variable exists in the variable stack.

    Parameters
    ----------
    name : str
        The dot-separated name of the variable to check.
    check_nested : bool, optional
        If True, raise an error if any of the keys (including the last key) are missing or
        if a non-dictionary object is encountered before the last key.
        If False, return False immediately if any key is missing or a non-dictionary object
        is encountered.
        Defaults to False.
    _parser_context : dict or None, optional
        Internal parser context.

    Returns
    -------
    bool
        True if the variable exists, False otherwise.
    """
    if not isinstance(name, str):
        raise TypeError(f"'name' must be of type str, got {type(name)}.")
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")

    variable_stack = _parser_context['variable_stack']
    var = variable_stack
    keys = name.split('.')

    for i, key in enumerate(keys):
        if isinstance(var, (dict, VariableStack)):
            try:
                var = var[key]
            except KeyError:
                if check_nested and i < len(keys) - 1:
                    path = '.'.join(keys[:i])
                    raise ValueError(f"Cannot index into non-dict object at '{path}'")
                else:
                    return False
        else:
            if check_nested:
                path = '.'.join(keys[:i])
                raise ValueError(f"Cannot index into non-dict object at '{path}'")
            else:
                return False

    return True
