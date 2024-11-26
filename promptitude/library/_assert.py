def assert_(value: bool, msg: str = None) -> None:
    """
    Assert a given value.

    Parameters
    ----------
    value : bool
        The value to check. If `False`, then an AssertionError will be raised.
    msg : str, optional
        The message of the assertion error.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the provided value is `False`.

    Examples
    --------
    Use within a guidance template:

    >>> program = guidance("{{assert False msg}}")
    >>> output = program(msg="This is an error")
    >>> print(isinstance(output._exception, AssertionError))
    True
    >>> print(str(output._exception))
    This is an error
    """
    if msg is None:
        assert value
    else:
        assert value, msg
