def contains(string: str, substring: str) -> bool:
    """
    Check if a string contains a substring.

    Parameters
    ----------
    string : str
        The string to search within.
    substring : str
        The substring to search for.

    Returns
    -------
    result : bool
        True if 'substring' is found within 'string', False otherwise.

    Raises
    ------
    TypeError
        If 'string' or 'substring' is not a string.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("{{#if (contains val 'substr')}}contains substring{{else}}does not contain substring{{/if}}")
    >>> output = program(val='this is a substr')
    >>> print(output)
    contains substring
    """
    if not isinstance(string, str):
        raise TypeError(f"'string' must be of type str, got {type(string)}.")
    if not isinstance(substring, str):
        raise TypeError(f"'substring' must be of type str, got {type(substring)}.")
    return substring in string
