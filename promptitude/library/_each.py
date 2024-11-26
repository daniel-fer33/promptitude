from typing import Iterable, Optional, Any, Dict

import asyncio
import builtins
import re

from .._utils import ContentCapture


async def each(
        iterable: Iterable,
        hidden: bool = False,
        parallel: bool = False,
        item_name: str = "this",
        start_index: int = 0,
        _parser_context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Iterate over an iterable and execute a block for each item.

    Parameters
    ----------
    iterable : Iterable
        The iterable to iterate over. Inside the block, each element will be available as `this` (or via `item_name`).
    hidden : bool, optional
        Whether to include the generated item blocks in future LLM context. Default is False.
    parallel : bool, optional
        If True, the items in the iterable are processed in parallel. Only compatible with hidden=True.
    item_name : str, optional
        The name of the variable to use for the current item in the iterable. Default is "this".
    start_index : int, optional
        The index from which to start iterating over the iterable. Default is 0.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is None.

    Returns
    -------
    Any
        The result of the executed block(s).

    Raises
    ------
    TypeError
        If 'iterable' is not actually iterable.
    ValueError
        If '_parser_context' is None.
        If 'parallel=True' and 'hidden=False'.

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance("Hello, {{name}}!{{#each names}} {{this}}{{/each}}")
    >>> output = program(name="Guidance", names=["Bob", "Sue"])
    >>> print(output)
    Hello, Guidance! Bob Sue
    """
    if _parser_context is None:
        raise ValueError("'_parser_context' cannot be None.")
    if not hidden and parallel:
        raise ValueError(
            "parallel=True is only compatible with hidden=True "
            "(since if hidden=False earlier items are context for later items)")

    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    if isinstance(iterable, dict):
        keys = builtins.list(iterable.keys())
        iterable_values = iterable.values()
    else:
        keys = None
        iterable_values = iterable

    # Ensure the iterable is actually iterable
    try:
        iter(iterable_values)
    except TypeError:
        raise TypeError("The #each command cannot iterate over a non-iterable value: " + str(iterable))

    out = []

    # Process items in parallel if requested and if hidden is True
    if parallel:
        # Set up the coroutines to call
        coroutines = []
        for i, item in enumerate(iterable_values):
            if i < start_index:  # Skip items before the start index
                continue
            context = {
                "@index": i,
                "@first": i == 0,
                "@last": i == len(iterable_values) - 1,
                item_name: item,
                "@raw_prefix": variable_stack["@raw_prefix"],  # Create a local copy of the prefix since we are hidden
                "@no_display": True
            }
            if keys is not None:
                context["@key"] = keys[i]
            variable_stack.push(context)
            coroutines.append(parser.visit(
                block_content,
                variable_stack.copy(),
                next_node=_parser_context.get("next_node"),
                next_next_node=_parser_context.get("next_next_node"),
                prev_node=_parser_context.get("prev_node")
            ))
            variable_stack.pop()

        await asyncio.gather(*coroutines)

        # for item_out in item_outs:

        #     # parser._trim_prefix(item_out)
        #     out.append(item_out)

        #     # check if the block has thrown a stop iteration signal
        #     if parser.caught_stop_iteration:
        #         parser.caught_stop_iteration = False
        #         break

    else:
        for i, item in enumerate(iterable_values):
            if i < start_index:
                continue
            context = {
                "@index": i,
                "@first": i == 0,
                "@last": i == len(iterable_values) - 1,
                item_name: item
            }
            if keys is not None:
                context["@key"] = keys[i]
            variable_stack.push(context)
            with ContentCapture(variable_stack, hidden) as new_content:
                new_content += await parser.visit(
                    block_content,
                    variable_stack,
                    next_node=_parser_context.get("next_node"),
                    next_next_node=_parser_context.get("next_next_node"),
                    prev_node=_parser_context.get("prev_node")
                )
                out.append(str(new_content))
            variable_stack.pop()

            # If we stopped executing then we need to dump our node text back out but with£
            # the start_index incremented to account for what we've already done
            if not parser.executing:
                updated_text = re.sub(
                    r"^({{~?#each.*?)(~?}})",
                    r"\1 start_index=" + str(i + 1) + r"\2",
                    _parser_context["parser_node"].text
                )
                variable_stack["@raw_prefix"] += updated_text
                break

            # Check if the block has signaled to stop iteration
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break

    # if not hidden:
    # return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    # if hidden:
    #     id = uuid.uuid4().hex
    #     l = len(out)
    #     out_str = "{{!--" + f"GMARKER_each_noecho_start_{not hidden}_{l}${id}$" + "--}}"
    #     for i, value in enumerate(out):
    #         if i > 0:
    #             out_str += "{{!--" + f"GMARKER_each_noecho_{not hidden}_{i}${id}$" + "--}}"
    #         out_str += value
    #     return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"


each.is_block = True
