from typing import Any, Dict, Union
from .._utils import ContentCapture


async def role(role_name: str, hidden: bool = False, _parser_context: Union[Dict, None] = None, **kwargs: Any) -> Any:
    """
    A chat role block.

    This function defines a block for a specific chat role, such as 'system', 'user', or 'assistant'. It can be used within guidance templates to specify messages from different roles.

    Parameters
    ----------
    role_name : str
        The name of the role (e.g., 'system', 'user', 'assistant').
    hidden : bool, optional
        Whether to include the role block in future LLM context. Default is False.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        The result of the role block execution.

    Raises
    ------
    TypeError
        If 'role_name' is not a string or 'hidden' is not a boolean.
    ValueError
        If '_parser_context' is None.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance(\"\"\"
    ... {{#role 'user'}}
    ... What is the weather?
    ... {{/role}}
    ... \"\"\")
    >>> output = program()
    """
    # Type checking
    if not isinstance(role_name, str):
        raise TypeError(f"'role_name' must be of type str, got {type(role_name)}.")
    if not isinstance(hidden, bool):
        raise TypeError(f"'hidden' must be of type bool, got {type(hidden)}.")
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")

    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # send the role-start special tokens
        new_content += parser.program.llm.role_start(role_name, **kwargs)

        # visit the block content
        new_content += await parser.visit(
            block_content,
            variable_stack,
            next_node=_parser_context["block_close_node"],
            prev_node=_parser_context["prev_node"],
            next_next_node=_parser_context["next_node"]
        )

        # send the role-end special tokens
        new_content += parser.program.llm.role_end(role_name)


role.is_block = True
