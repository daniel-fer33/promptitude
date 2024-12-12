def raise_(msg: str = None, **kwargs) -> None:
    """
    Raise an Exception with the provided message.

    Parameters
    ----------
    msg : str, optional
        The message of the exception.

    **kwargs
        Additional keyword arguments to format the msg.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Always raises an exception with the provided message.

    Examples
    --------
    Use within a guidance template:

    >>> program = guidance("{{raise msg}}")
    >>> output = program(msg="This is an error")
    >>> print(isinstance(output._exception, Exception))
    True
    >>> print(str(output._exception))
    This is an error

    >>> program = guidance("{{raise msg}}")
    >>> output = program(msg="Error code: {code}", code=404)
    >>> print(str(output._exception))
    Error code: 404
    """
    if kwargs:
        msg = msg.format(**kwargs)
    raise Exception(msg)
