from typing import Any, Optional, Dict

from ._if import if_


async def unless(value: Any, _parser_context: Optional[Dict] = None) -> Any:
    """
    Execute a block only if the given condition is False.

    This is the inverse of the 'if' block and can be used in guidance templates
    to conditionally render content when a condition is not met.

    Parameters
    ----------
    value : Any
        The condition to evaluate. If False, the block executes.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    Any
        The result of the 'if' block execution with the inverted condition.

    Raises
    ------
    TypeError
        If '_parser_context' is not a dict or None.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Answer: {{#unless flag}}Yes{{/unless}}")
    >>> output = program(flag=False)
    >>> print(str(output))
    Answer: Yes
    """
    if _parser_context is not None and not isinstance(_parser_context, Dict):
        raise TypeError(f"'_parser_context' must be a dict or None, got {type(_parser_context)}.")
    return await if_(value, invert=True, _parser_context=_parser_context)


unless.is_block = True
