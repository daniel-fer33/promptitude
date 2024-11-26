from typing import Any, Union, Dict

from .._utils import ContentCapture


async def block(name: Union[str, None] = None, hidden: bool = False, _parser_context: Union[Dict, None] = None) -> Any:
    """
    A generic block-level element.

    This is useful for naming or hiding blocks of content.

    Parameters
    ----------
    name : str or None, optional
        The name of the block. A variable with this name will be set with the generated block content.
        Default is None.
    hidden : bool, optional
        Whether to include the generated block content in future LLM context. Default is False.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    Any
        The result of the block execution.

    Raises
    ------
    TypeError
        If 'hidden' is not a boolean.
        If 'name' is not a string or None.
    ValueError
        If '_parser_context' is None.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("This is a block: {{#block name='my_block'}}text inside block{{/block}}")
    >>> output = program()
    >>> print(output["my_block"])
    text inside block
    """
    if not isinstance(hidden, bool):
        raise TypeError(f"'hidden' must be of type bool, got {type(hidden)}.")
    if name is not None and not isinstance(name, str):
        raise TypeError(f"'name' must be of type str or None, got {type(name)}.")
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")

    parser = _parser_context.get('parser')
    variable_stack = _parser_context.get('variable_stack')

    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # visit the block content
        new_content += await parser.visit(
            _parser_context.get('block_content', [])[0],
            variable_stack,
            next_node=_parser_context.get("next_node"),
            next_next_node=_parser_context.get("next_next_node"),
            prev_node=_parser_context.get("prev_node")
        )

        # set the variable if needed
        if name is not None:
            variable_value = str(new_content)
            variable_stack[name] = variable_value


block.is_block = True
