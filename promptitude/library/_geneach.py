from typing import Optional, Union, Dict, Any, List

import re

from .._utils import ContentCapture


async def geneach(
    list_name: str,
    stop: Optional[Union[str, List[str]]] = None,
    max_iterations: int = 100,
    min_iterations: int = 0,
    num_iterations: Optional[int] = None,
    hidden: bool = False,
    join: str = "",
    single_call: bool = False,
    single_call_temperature: float = 0.0,
    single_call_max_tokens: int = 500,
    single_call_top_p: float = 1.0,
    _parser_context: Optional[Dict[str, Any]] = None,
):
    """
    Generate a potentially variable length list of items using the LLM.

    Parameters
    ----------
    list_name : str
        The name of the variable to save the generated list to.
    stop : str or list of str, optional
        A string or list of strings that will stop the generation of the list. For example, if `stop="</ul>"`,
        then the list will be generated until the first `"</ul>"` is generated.
    max_iterations : int, optional
        The maximum number of items to generate. Default is `100`.
    min_iterations : int, optional
        The minimum number of items to generate. Default is `0`.
    num_iterations : int, optional
        The exact number of items to generate (this overrides `max_iterations` and `min_iterations`).
    hidden : bool, optional
        If `True`, the generated list items will not be added to the LLM's input context. This means that each
        item will be generated independently of the others. Note that if you use `hidden=True`, you must also
        set `num_iterations` to a fixed number (since without adding items to the context, there is no way for the
        LLM to know when to stop on its own).
    join : str, optional
        A string to join the generated items with. Default is an empty string.
    single_call : bool, optional
        This is an option designed to make loop generation more convenient for LLMs that don't support guidance
        acceleration. If `True`, the LLM will be called once to generate the entire list. This only works if the
        LLM has already been prompted to generate content that matches the format of the list. After the single
        call, the generated list variables will be parsed out of the generated text using a regex. (Note that only
        basic template tags are supported in the list items when using `single_call=True`). Default is `False`.
    single_call_temperature : float, optional
        Only used with `single_call=True`. The temperature to use when generating the list items in a single call. Default is `0.0`.
    single_call_max_tokens : int, optional
        Only used with `single_call=True`. The maximum number of tokens to generate when generating the list items. Default is `500`.
    single_call_top_p : float, optional
        Only used with `single_call=True`. The `top_p` to use when generating the list items in a single call. Default is `1.0`.
    _parser_context : dict or None, optional
        Internal parser context (used internally). Default is `None`.

    Returns
    -------
    None

    Examples
    --------
    Use within a guidance template:

    >>> from promptitude import guidance
    >>> program = guidance('''
    ... <instructions>Generate a list of three names</instructions>
    ... <list>{{#geneach 'names' num_iterations=3}}
    ... <item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>
    ... ''')
    >>> output = program()
    >>> print(output["names"])
    ['Name1', 'Name2', 'Name3']
    """
    block_content = _parser_context["block_content"]
    parser = _parser_context["parser"]
    variable_stack = _parser_context["variable_stack"]
    # parser_prefix = _parser_context["parser_prefix"]
    parser_node = _parser_context["parser_node"]

    # assert len(block_content) == 1
    assert not (hidden and single_call), "Cannot use hidden=True and single_call together"
    assert isinstance(list_name, str), "Must provide a variable name to save the generated list to"
    assert not hidden or num_iterations is not None, "Cannot use hidden=True and variable length iteration together yet..."
    echo = not hidden

    # num_iterations has priority over max_iterations if they are both set
    if num_iterations is not None:
        max_iterations = num_iterations
        min_iterations = num_iterations

    if max_iterations is None:
        max_iterations = 1e10

    # give the list a default name
    # if list_name is None:
    #     list_name = 'generated_list'

    # if stop is None then we use the text of the node after the generate command
    # if stop is None:

    #     next_text = next_node.text if next_node is not None else ""
    #     prev_text = prev_node.text if prev_node is not None else ""

    #     # auto-detect quote stop tokens
    #     quote_types = ['"', "'", "'''", '"""', "`"]
    #     for quote_type in quote_types:
    #         if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
    #             stop = quote_type
    #             break
                
    #     # auto-detect XML tag stop tokens
    #     if stop is None:
    #         m = re.match(r"^\s*(</[^>]+>)", next_text, re.DOTALL) #next_text.startswith(end_tag)
    #         if m is not None:
    #             stop = m.group(1)
            
    #         m = re.match(r"^\s*(<|im_end|>)", next_text, re.DOTALL) #next_text.startswith(end_tag)
    #         if m is not None:
    #             stop = "<|im_end|>"
            
    #         if next_text != "":
    #             stop = next_text

    out = []
    partial_out = ""
    
    # convert stop strings to tokens
    if stop is not False:
        if stop is None:
            max_stop_tokens = 2
        else:
            max_stop_tokens = max([len(parser.program.llm.encode(s)) for s in stop]) + 2

    if not single_call:
        for i in range(max_iterations):
            
            # capture the content generated by the block
            with ContentCapture(variable_stack, hidden) as new_content:
                
                # set the variables for this iteration
                variable_stack.push({
                    "@index": i,
                    "@first": i == 0,
                    "this": {}
                })

                # add the join string if we are not on the first iteration
                if i > 0 and join != "":
                    new_content += join

                # visit the block content
                new_content += await parser.visit(
                    block_content,
                    variable_stack,
                    next_node=_parser_context["next_node"],
                    next_next_node=_parser_context["next_next_node"],
                    prev_node=_parser_context["prev_node"]
                )
            
                # update the list variable (we do this each time we get a new item so that streaming works)
                block_variables = variable_stack.pop()["this"]
                variable_stack[list_name] = variable_stack.get(list_name, []) + [block_variables]

                # stop if we are not executing anymore            
                if not parser.executing:

                    # make any unfinished `this.` references point to the last (unfinished) item
                    new_content.inplace_replace("this.", list_name+"[-1].")
                    break

            # check if the block has thrown a stop iteration signal
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break

            # we run a quick generation to see if we have reached the end of the list (note the +2 tokens is to help be tolorant to whitespace)
            if stop is not False and i >= min_iterations and i < max_iterations:
                try:
                    gen_obj = await parser.llm_session(variable_stack["@prefix"], stop=stop, max_tokens=max_stop_tokens, temperature=0, cache_seed=0)
                except Exception:
                    raise Exception(f"Error generating stop tokens for geneach loop. Perhaps you are outside of role tags (assistant/user/system/function)? If you don't want the loop to check for stop tokens, set stop=False or set num_iterations.")
                if gen_obj["choices"][0]["finish_reason"] == "stop":
                    break
    
    # TODO: right now single_call is a bit hacky, we should make it more robust to rich loop item template structures
    else: # if single_call
        # create a pattern to match each item
        pattern = re.sub(
            r'{{gen [\'"]([^\'"]+)[\'"][^}]*}}',
            lambda x: r"(?P<"+_escape_group_name(x.group(1))+">.*?)",
            block_content.text
        )

        # fixed prefixes can be used if we know we have at least one iteration
        if min_iterations > 0:
            # find what part of the pattern is fixed before the first generation
            fixed_prefix = re.match(r"^(.*)\(\?P\<", pattern, flags=re.DOTALL)[0][:-4]
            fixed_prefix = fixed_prefix.replace(r"{{@index}}", "0") # TODO: this is a bit hacky
        else:
            fixed_prefix = ""

        # assume the LLM will also generate whatever interpolations are in the pattern
        pattern = re.sub(r"{{(.*?)}}", lambda x: r"(?P<" + _escape_group_name(x.group(1)) + ">.*?)", pattern)

        # generate the looped content
        if single_call_temperature > 0:
            cache_seed = parser.program.cache_seed
            parser.program.cache_seed += 1
        else:
            cache_seed = 0
        gen_stream = await parser.llm_session(variable_stack["@raw_prefix"]+fixed_prefix, stop=stop, max_tokens=single_call_max_tokens, temperature=single_call_temperature, top_p=single_call_top_p, cache_seed=cache_seed, stream=True)
        generated_value = fixed_prefix
        num_items = 0
        data = []
        for gen_obj in gen_stream:
            generated_value += gen_obj["choices"][0]["text"]


            # parse the generated content (this assumes the generated content is syntactically correct)
            matches = re.finditer(pattern, generated_value)
            for m in matches:#"{{!--" + f"GMARKER_START_{name}${node_text}$}}{out}{{!--GMARKER_END_{name}$$" + "}}"
                
                # consume the generated value up to the match
                generated_value = generated_value[m.end():]

                # get the variables that were generated
                match_dict = m.groupdict()
                if "this" in match_dict:
                    next_item = match_dict["this"]
                else:
                    d = {}
                    for k in match_dict:
                        k = _unescape_group_name(k)
                        if k.startswith("this."):
                            d[k[5:]] = match_dict[k].strip()
                    next_item = d

                # update the list variable (we do this each time we get a new item so that streaming works)
                variable_stack[list_name] = variable_stack.get(list_name, []) + [next_item]

                # recreate the output string with format markers added
                item_out = re.sub(
                    r"{{(?!~?gen)(.*?)}}",
                    lambda x: match_dict[_escape_group_name(x.group(1))],
                    block_content.text
                )
                item_out = re.sub(
                    r"{{gen [\'\"]([^\'\"]+)[\'\"][^}]*}}",
                    lambda x: "{{!--GMARKER_START_gen$"+x.group().replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")+"$--}}"+match_dict[_escape_group_name(x.group(1))]+"{{!--GMARKER_END_gen$$--}}",
                    item_out
                )
                variable_stack["@raw_prefix"] += "{{!--GMARKER_each$$--}}" + item_out # marker and content of the item
                num_items += 1
                # out.append(item_out)

                # if we have hit the max iterations, stop the LLM
                if num_items >= max_iterations:
                    gen_stream.close()
    
    # partial_output("{{!--GMARKER_each$$--}}") # end marker

    # parser.get_variable(list, [])
    #parser.set_variable(list_name, parser.get_variable(list_name, default_value=[]) + data)
   
    # if we have stopped executing, we need to add the loop to the output so it can be executed later
    if not parser.executing:
        variable_stack["@raw_prefix"] += parser_node.text

    # return ""
    
    # if echo:
    #     return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    # else:
    #     id = uuid.uuid4().hex
    #     l = len(out)
    #     out_str = prefix + "{{!--" + f"GMARKER_each_noecho_start_{echo}_{l}${id}$" + "--}}"
    #     for i, value in enumerate(out):
    #         if i > 0:
    #             out_str += "{{!--" + f"GMARKER_each_noecho_{echo}_{i}${id}$" + "--}}"
    #         out_str += value
    #     return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

    #     # return "{{!--GMARKER_each_noecho$$}}" + "{{!--GMARKER_each_noecho$$}}".join(out) + "{{!--GMARKER_each_noecho$$}}"


geneach.is_block = True


def _escape_group_name(name: str) -> str:
    return name.replace("@", "_AT_").replace(".", "_DOT_")


def _unescape_group_name(name: str) -> str:
    return name.replace("_AT_", "@").replace("_DOT_", ".")
