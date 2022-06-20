#! /usr/bin/env /Users/mlevental/dev_projects/hls_paper/venv/bin/python
import argparse
import sys

import pygments.cmdline as _cmdline
from pygments.lexer import RegexLexer, include, words
from pygments.token import Name, String, Number, Punctuation, Whitespace, Comment, Keyword, Operator


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='lexer', type=str)
    opts, rest = parser.parse_known_args(args[1:])
    if opts.lexer == 'mlir':
        args = [__file__, '-l', __file__ + ':MLIRLexer', '-x', *rest]
    _cmdline.main(args)


# '\\([^)>]*\\)\\s*->\\s*\\([^)>]*\\)'

class MLIRLexer(RegexLexer):
    #: optional Comment or Whitespace
    string = r'"[^"]*?"'
    identifier = r'([-a-zA-Z$._][\w\-$.]*|' + string + ')'
    block_label = r'(' + identifier + r'|(\d+))'

    tokens = {
        'root': [
            include('whitespace'),

            # Before keywords, because keywords are valid label names :(...
            (block_label + r'\s*:', Name.Label),

            include('keyword'),

            (r'%' + identifier, Name.Variable),
            (r'@' + identifier, Name.Variable.Global),
            (r'%\d+', Name.Variable.Anonymous),
            (r'@\d+', Name.Variable.Global),
            (r'#\d+', Name.Variable.Global),
            (r'#[a-z]+', Name.Variable.Global),
            (r'c?' + string, String),
            # (r'(:|::|->)', Operator.Word),
            (r'prim', Keyword),
            (r'aten', Keyword),

            (r'0[xX][a-fA-F0-9]+', Number),
            (r'-?\d+(?:[.]\d+)?(?:[eE][-+]?\d+(?:[.]\d+)?)?', Number),

            (r'[=<>{}\[\]()*.,#]|x\b', Punctuation),
            (r'\([^)>]*\)\s*->\s*\([^)>]*\)', Operator.Word),
        ],
        'whitespace': [
            (r'(\n|\s+)+', Whitespace),
            (r';.*?\n', Comment)
        ],
        'keyword': [
            # Regular keywords
            (words((
                'memref', 'for', 'parallel', 'scf', 'step', 'arith', 'addi', 'get_global', 'relu'
                'func', 'addf', 'mulf', 'graph', 'subf', 'expf', 'divf',
                'Constant', 'value', 'strides', 'requires_grad',
                'device', 'cpu', 'conv2d', 'torch', 'vtensor', 'ListConstruct',
                'affine', 'copy', 'literal', 'linalg', 'cmpi', 'fill', 'ins', 'outs', 'init_tensor',
                'dilations', 'conv_2d_nchw_fchw', 'affine_map', 'map', 'apply', 'constant', 'none',
                'eq', 'load', 'store', 'list', 'return', 'to', 'alloca', 'alloc', 'global', 'true', 'int', 'llvm',
            ),
                suffix=r'\b'), Keyword),

            # Types
            (words(('void', 'half', 'bfloat', 'float', 'double', 'fp128', 'Tensor', 'NoneType', 'Float', 'tensor',
                    'x86_fp80', 'ppc_fp128', 'label', 'metadata', 'x86_mmx', 'xf16',
                    'x86_amx', 'token', 'index', 'f32')),
             Keyword.Type),

            (
                r"(((\?|[1-9][0-9]*)\s*x\s*)*)(i[1-9][0-9]*|f16|bf16|f32|f64|u8|ui32|si32|!quant\.uniform|vector|dense|tensor|memref|!)\b",
                Keyword.Type),

        ]
    }


if __name__ == '__main__':
    main(sys.argv)
