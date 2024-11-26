from typing import Any, Union, Dict

from ._role import role


async def system(hidden: bool = False, _parser_context: Union[Dict, None] = None, **kwargs: Any) -> Any:
    """
    A chat role block for the 'system' role.

    This is a shorthand for ``{{#role 'system'}}...{{/role}}`` within guidance templates.

    Parameters
    ----------
    hidden : bool, optional
        Whether to include the system block in future LLM context. Default is False.
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
    ... {{#system}}
    ... You are a helpful assistant.
    ... {{/system}}
    ... {{#user}}
    ... What is the capital of France?
    ... {{/user}}
    ... {{#assistant}}
    ... {{gen 'output'}}
    ... {{/assistant}}
    ... ''')
    >>> output = program()
    >>> print(output["output"])
    """
    if not isinstance(hidden, bool):
        raise TypeError(f"'hidden' must be of type bool, got {type(hidden)}.")
    return await role(role_name="system", hidden=hidden, _parser_context=_parser_context, **kwargs)


system.is_block = True
