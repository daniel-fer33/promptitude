from .._utils import ContentCapture
from .._program_executor import parse_program_string


async def parse(string, name=None, hidden=False, hidden_prefix=False, _parser_context=None):
    ''' Parse a string as a guidance program.

    This is useful for dynamically generating and then running guidance programs (or parts of programs).

    Parameters
    ----------
    string : str
        The string to parse.
    name : str
        The name of the variable to set with the generated content.
    hidden : bool, optional
        If `True`, the generated content is hidden; otherwise, it is visible. Defaults to `False`.
    hidden_prefix : bool, optional
        If `True`, the previous content (prefix) is hidden; otherwise, it is visible. Defaults to `False`.
    '''

    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']
    prev_raw_prefix = variable_stack["@raw_prefix"]

    if hidden_prefix:
        # Hide previous prefix
        variable_stack["@raw_prefix"] = ""

    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # parse and visit the given string
        subtree = parse_program_string(string)
        new_content += await parser.visit(subtree, variable_stack)

        # save the content in a variable if needed
        if name is not None:
            variable_stack[name] = str(new_content)

    if hidden_prefix:
        # Recover state
        new_raw_prefix = variable_stack["@raw_prefix"]
        variable_stack["@raw_prefix"] = prev_raw_prefix
        if not hidden:
            variable_stack["@raw_prefix"] += new_raw_prefix
