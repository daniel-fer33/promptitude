def assert_(value, msg=None):
    ''' Assert a given value.

    Parameters
    ----------
    value : bool
        The value to check. If `False` then an AssertionError will be raised.
    msg : srt
        Message of the assertion error.
    '''
    if msg is None:
        assert value
    else:
        assert value, msg
