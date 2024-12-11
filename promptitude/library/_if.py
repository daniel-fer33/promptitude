from typing import Any, Union, Dict


async def if_(value: Any, *, invert: bool = False, _parser_context: Union[Dict, None] = None) -> str:
    """
    Standard if/else statement.

    Parameters
    ----------
    value : Any
        The value to check. If evaluated as `True`, then the first block will be executed; otherwise, the subsequent blocks
        (after `{{elif}}` or `{{else}}`) will be evaluated.
    invert : bool, optional
        [DEPRECATED] If `True`, the value will be inverted before checking. Default is False.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    str
        The result of the executed block.

    Raises
    ------
    ValueError
        If unexpected block content is encountered, or if `_parser_context` is `None`.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> prompt = guidance("Answer: {{#if flag}}Yes{{else}}No{{/if}}")
    >>> output = prompt(flag=True)
    >>> print(output)
    Answer: Yes

    >>> output = prompt(flag=False)
    >>> print(output)
    Answer: No
    """
    if _parser_context is None:
        raise ValueError("_parser_context cannot be None")

    block_content = _parser_context['block_content']
    variable_stack = _parser_context['variable_stack']
    parser = _parser_context['parser']

    if len(block_content) % 2 != 1:
        raise ValueError("Unexpected number of blocks for `if` command: " + str(len(block_content)))

    # parse the first block
    if invert:
        value = not value
    if value:
        return await parser.visit(block_content[0], variable_stack)

    # parse the rest of the blocks
    for i in range(1, len(block_content), 2):

        # elif block
        if block_content[i][0] == "elif":
            elif_condition = await parser.visit(block_content[i][1], variable_stack)
            if elif_condition.value:
                return await parser.visit(block_content[i + 1], variable_stack)

        # else block
        elif block_content[i][0] == "else":
            return await parser.visit(block_content[i + 1], variable_stack)

        else:
            raise ValueError("Unexpected block content separator for `if` command: " + block_content[i].text)
    return ""


if_.is_block = True
