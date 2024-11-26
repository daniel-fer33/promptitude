def strip(string: str) -> str:
    """
    Strip whitespace from the beginning and end of the given string.

    Parameters
    ----------
    string : str
        The string to strip.

    Returns
    -------
    result : str
        The stripped string.

    Raises
    ------
    TypeError
        If the input is not a string.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("{{strip ' this is '}}")
    >>> output = program()
    >>> print(output)
    this is
    """
    if not isinstance(string, str):
        raise TypeError(f"'string' must be of type str, got {type(string)}.")
    return string.strip()
