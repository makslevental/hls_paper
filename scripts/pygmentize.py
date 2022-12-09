#! /usr/bin/env /Users/Max/dev_projects/hls_paper/venv/bin/python
# \renewcommand{\MintedPygmentize}{/Users/mlevental/dev_projects/hls_paper/scripts/pygmentize.py}
import argparse
import re
import sys

import pygments.cmdline as _cmdline
from pygments.lexer import RegexLexer, include, bygroups, using, this
from pygments.token import *
from pygments.token import _TokenType


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", dest="lexer", type=str)
    opts, rest = parser.parse_known_args(args[1:])
    if opts.lexer == "mlir":
        args = [__file__, "-l", __file__ + ":MLIRLexer", "-x", *rest]
    _cmdline.main(args)


PRINT_ERRORS = False


class MLIRLexer(RegexLexer):
    name = "mlir"
    flags = re.MULTILINE

    def get_tokens_unprocessed(self, text, stack=("root",)):
        pos = 0
        tokendefs = self._tokens
        statestack = list(stack)
        statetokens = tokendefs[statestack[-1]]
        while 1:
            for rexmatch, action, new_state in statetokens:
                m = rexmatch(text, pos)
                if m:
                    if action is not None:
                        if type(action) is _TokenType:
                            yield pos, action, m.group()
                        else:
                            yield from action(self, m)
                    pos = m.end()
                    if new_state is not None:
                        # state transition
                        if isinstance(new_state, tuple):
                            for state in new_state:
                                if state == "#pop":
                                    if len(statestack) > 1:
                                        statestack.pop()
                                elif state == "#push":
                                    statestack.append(statestack[-1])
                                else:
                                    statestack.append(state)
                        elif isinstance(new_state, int):
                            # pop, but keep at least one state on the stack
                            # (random code leading to unexpected pops should
                            # not allow exceptions)
                            if abs(new_state) >= len(statestack):
                                del statestack[1:]
                            else:
                                del statestack[new_state:]
                        elif new_state == "#push":
                            statestack.append(statestack[-1])
                        else:
                            assert False, "wrong state def: %r" % new_state
                        statetokens = tokendefs[statestack[-1]]
                    break
            else:
                # We are here only if all state tokens have been considered
                # and there was not a match on any of them.
                try:
                    if text[pos] == "\n":
                        # at EOL, reset state to "root"
                        statestack = ["root"]
                        statetokens = tokendefs["root"]
                        yield pos, Text, "\n"
                        pos += 1
                        continue
                    # yield pos, Error if PRINT_ERRORS else Text, text[pos]
                    yield pos, Text, text[pos]
                    pos += 1
                except IndexError:
                    break

    tokens = {
        "whitespace": [
            (r"\n", Text),
            (r"\s+", Text),
            (r"\\\n", Text),  # line continuation
        ],
        "root": [
            include("whitespace"),
            include("string"),
            include("comment"),
            include("top_level_entity"),
        ],
        "comment": [(r"//.*$", Comment)],
        "number": [
            (
                r"(\W)?([0-9]+\.[0-9]*)([eE][+-]?[0-9]+)?",
                Literal.Number,
            ),
            (
                r"([\W])?(0x[0-9a-zA-Z]+)",
                bygroups(Punctuation, Literal.Number),
            ),
            (r"([\Wx])?([0-9]+)", bygroups(Punctuation, Literal.Number)),
        ],
        "string": [
            (r'"', Punctuation, "inside_string"),  # Opening quote
        ],
        "inside_string": [
            (r"\n", Text),
            (r'\\[nt"]', String.Escape),
            (r'"', Punctuation, "#pop"),  # Closing quot
            (r".", String),  # String content
        ],
        "top_level_entity": [
            include("attribute_alias_def"),
            include("type_alias_def"),
            include("operation_body"),
            include("operation"),
        ],
        "attribute_alias_def": [
            (
                r"^(\s*)(#\w+)\b(\s+)(=)(\s+)",
                bygroups(
                    Whitespace, Name.Constant, Whitespace, Punctuation, Whitespace
                ),
                "attribute_value",
            )
        ],
        "type_alias_def": [
            (
                r"^(\s*)(!\w+)\b(\s+)(=)(\s+)",
                bygroups(Whitespace, Name.Type, Whitespace, Punctuation, Whitespace),
            )
        ],
        "operation": [
            (
                r"^(\s*)(%[%\w:,\s]+)(\s+)(=)(\s+)([\w.$\-]+)\b",
                bygroups(
                    Whitespace,
                    using(this, state="ssa_value"),
                    Whitespace,
                    Punctuation,
                    Whitespace,
                    Name.Variable,
                ),
            ),
            (r"^\s*([\w.$\-]+)\b(?=[^<:])", Name.Variable),
        ],
        "operation_body": [
            include("operation"),
            (r"{\s*(?=%|\/|\^)", Punctuation, "inside_region_body_or_attr_dict"),
            (r"{\s*(?=[^}]*$)", Punctuation, "inside_region_body_or_attr_dict"),
            (r"{\s*(?=%)", Punctuation, "inside_region_body_or_attr_dict"),
            (r"{\s*(?=.*$)", Punctuation, "inside_region_body_or_attr_dict"),
            include("comment"),
            include("ssa_value"),
            include("block"),
            include("attribute_value"),
            include("bare_identifier"),
        ],
        "inside_region_body_or_attr_dict": [
            include("operation_body"),
            (r"}", Punctuation, "#pop"),
        ],
        "attribute_dictionary_body": [
            include("comment"),
            include("string"),
            include("attribute_value"),
            (r"(%)?\b([\w.\-$:0-9]+)\b\s*(?=[=,}])", Name.Variable),
            (r"}", Punctuation, "#pop"),
        ],
        #
        "attribute_value": [
            include("number"),
            (r"\b(false|true|unit)\b", Name.Constant),
            (r"(@[\w+$\-.]*)", Name.Function),
            (r"#[\w$\-.]+\b", Name.Constant),
            (
                r"\b(affine_map|affine_set)(\s*)(<)",
                bygroups(Name.Constant, Whitespace, Punctuation),
                "inside_affine_map_set",
            ),
            (
                r"\b(dense|opaque|sparse)(\s*)(<)",
                bygroups(Name.Constant, Whitespace, Punctuation),
                "inside_dense_opaque_sparse",
            ),
            (
                r"\[",
                Punctuation,
                "inside_square_brackets",
            ),
            (
                r"{",
                Punctuation,
                "inside_attribute_dictionary_body",
            ),
            (r"(@[\w+$\-.]*)", Name.Function),
            (
                r"(#[\w$\-.]+)(<)",
                bygroups(Name.Constant, Punctuation),
                "inside_hash",
            ),
            include("type_value"),
            (
                r"<",
                Punctuation,
                "inside_angle",
            ),
        ],
        "inside_affine_map_set": [
            (r"\b(ceildiv|floordiv|mod|symbol)\b", Name.Function),
            (r"\b([\w\.\$\-]+)\b", Name.Variable),
            include("number"),
            (r"\)>", Punctuation, "#pop"),
        ],
        "inside_dense_opaque_sparse": [
            include("attribute_value"),
            (r">", Punctuation, "#pop"),
        ],
        "inside_square_brackets": [
            include("attribute_value"),
            include("operation_body"),
            (r"\]", Punctuation, "#pop"),
        ],
        "inside_attribute_dictionary_body": [
            include("attribute_dictionary_body"),
            include("operation_body"),
            (r"\]", Punctuation, "#pop"),
        ],
        "inside_hash": [
            include("attribute_value"),
            (r"->|>=", Punctuation),
            include("bare_identifier"),
            (r">", Punctuation, "#pop"),
        ],
        "inside_angle": [
            include("attribute_value"),
            include("bare_identifier"),
            (r">", Punctuation, "#pop"),
        ],
        "type_value": [
            (r"(\![\w\$\-\.]+)<", Name.Type, "inside_hash"),
            (r"\![\w\$\-\.]+\b", Name.Type),
            (r"(complex|memref|tensor|tuple|vector)<", Name.Type, "inside_type"),
            (
                r"(x)?(bf16|f16|f32|f64|f80|f128|index|none|[us]?i[0-9]+)",
                bygroups(Punctuation, Name.Type),
            ),
        ],
        "inside_type": [
            (r"([?x0-9\[\]]+)", bygroups(using(this, state="number"))),
            include("attribute_value"),
            (r"->|>=", Punctuation),
            include("bare_identifier"),
            (r">", Punctuation, "#pop"),
        ],
        #
        "ssa_value": [(r"%[\w.$:#]+", Name.Variable)],
        "bare_identifier": [(r"\b([\w.$\-]+)\b", Keyword)],
        "block": [(r"\^[\w\d_$.-]+", Keyword.Control)],
    }


if __name__ == "__main__":
    main(sys.argv)

