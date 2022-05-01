#!/usr/bin/env python
import sys

from pandocfilters import toJSONFilter, RawInline

incomment = False


def latex(s):
    return RawInline('latex', s)


def comment(k, v, fmt, meta):
    global incomment
    if k == 'Span':
        fmt, s = v
        _, comment_start_end, rest = fmt
        if comment_start_end == ['comment-start']:
            incomment = True
            print("comment_start", s, file=sys.stderr)
            res = [
                latex('\\commnt{'),
                *s,
                latex('}{'),
            ]
            return res

        elif comment_start_end == ['comment-end']:
            print("comment_end", s, file=sys.stderr)
            incomment = False
            return [latex('}')]


import re


def one_sentence_per_line(fp):
    s = open("sketch.md").read()
    s = s.replace(".\n\n", "-*").replace("\n", " ").replace(". ", ".\n").replace("-*", "\n\n")
    s = s.replace("`", "").replace("{=latex}", "")
    print(s)

if __name__ == "__main__":
    # toJSONFilter(comment)
    one_sentence_per_line("sketch.md")
