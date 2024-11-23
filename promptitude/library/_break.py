def break_() -> None:
    """
    Break out of the current loop.

    This function is useful for breaking out of a loop early in guidance templates,
    typically used inside an `{{#if ...}}...{{/if}}` block.

    Raises
    ------
    StopIteration
        Raised to signal the loop to stop.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance('''
    ... {{#each list}}
    ... {{this}}
    ... {{#if (equal this 5)}}{{break}}{{/if}}
    ... {{/each}}
    ... ''')
    >>> output = program(list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> print(output.text)
    1
    2
    3
    4
    5
    """
    raise StopIteration()