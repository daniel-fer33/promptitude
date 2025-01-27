from typing import Any


def contains(container: Any, item: Any) -> bool:
    """
    Check if ``item`` is present in ``container``.

    Now supports any built-in type that implements the ``in`` operator,
    such as strings, lists, or dictionaries.

    Parameters
    ----------
    container : Any
        The container to search within.
    item : Any
        The item to search for.

    Returns
    -------
    result : bool
        True if ``item`` is found within ``container``, False otherwise.

    Raises
    ------
    TypeError
        If ``container`` does not support the membership test (the ``in`` operator).
    """
    if not hasattr(container, "__contains__"):
        raise TypeError(f"'container' does not support membership test: {type(container)}")

    return item in container
