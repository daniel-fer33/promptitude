from typing import Any, Optional, Dict

from .._utils import ContentCapture
from .._program_executor import parse_program_string


async def parse(string: str, name: Optional[str] = None, hidden: bool = False, hidden_prefix: bool = False,
                _parser_context: Optional[Dict[str, Any]] = None) -> None:
    """
    Parse a string as a guidance program.

    This is useful for dynamically generating and then running guidance programs (or parts of programs).

    Parameters
    ----------
    string : str
        The string to parse as a guidance program.
    name : str, optional
        The name of the variable to set with the generated content. If `None`, the content is not saved to a variable.
        Defaults to `None`.
    hidden : bool, optional
        If `True`, the generated content is hidden; otherwise, it is visible. Defaults to `False`.
    hidden_prefix : bool, optional
        If `True`, the previous content (prefix) is hidden before parsing; otherwise, it is visible. Defaults to `False`.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Defaults to `None`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `_parser_context` is `None`.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("This is parsed: {{parse template}}")
    >>> output = program(template="My name is {{name}}", name="Bob")
    >>> print(str(output))
    This is parsed: My name is Bob
    """
    if _parser_context is None:
        raise ValueError("_parser_context cannot be None")

    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']
    prev_raw_prefix = variable_stack["@raw_prefix"]

    if hidden_prefix:
        # Hide previous prefix
        variable_stack["@raw_prefix"] = ""

    # Capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:
        # Parse and visit the given string
        subtree = parse_program_string(string)
        new_content += await parser.visit(subtree, variable_stack)

        # Save the content in a variable if needed
        if name is not None:
            variable_stack[name] = str(new_content)

    if hidden_prefix:
        # Recover state
        new_raw_prefix = variable_stack["@raw_prefix"]
        variable_stack["@raw_prefix"] = prev_raw_prefix
        if not hidden:
            variable_stack["@raw_prefix"] += new_raw_prefix


parse.is_block = True
