from typing import Any, Union, Dict

from ._role import role


async def user(hidden: bool = False, _parser_context: Union[Dict, None] = None, **kwargs: Any) -> Any:
    """
    A chat role block for the 'user' role.

    This is a shorthand for ``{{#role 'user'}}...{{/role}}`` within guidance templates.

    Parameters
    ----------
    hidden : bool, optional
        Whether to include the user block in future LLM context. Default is False.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.
    **kwargs : Any
        Additional keyword arguments to pass to the role block.

    Returns
    -------
    Any
        The result of the role block execution.

    Raises
    ------
    TypeError
        If 'hidden' is not a boolean.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance('''
    ... {{#user}}
    ... {{gen 'input' save_prompt='prompt'}}
    ... {{/user}}
    ... ''')
    >>> output = program()
    >>> print(output["input"])
    """
    if not isinstance(hidden, bool):
        raise TypeError(f"'hidden' must be of type bool, got {type(hidden)}.")
    return await role(role_name="user", hidden=hidden, _parser_context=_parser_context, **kwargs)


user.is_block = True
