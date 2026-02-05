# pattern_parser.py
# 目的: 文節ベースパターン（ギャップ [G{m,n}] を含む）を Lark でパースし、AST を生成する
#
# 前提:
# - grammar.lark は最新版（parallel_chain / mod_parallel_count / block_repeat の仕様反映）
# - pattern_nodes.py は最新版（GapNode / ModifierParallelNode(count対応) を含む）

from lark import Lark, Transformer, v_args
from pattern_nodes import (
    PatternNode,
    SequenceNode,
    VariableNode,
    LiteralNode,
    GapNode,
    ParallelNode,
    ModifierRepeatNode,
    ModifierParallelNode,
    ModifierSingleNode,
    ModifierBlockRepeatNode,
)

import os

# Ensure grammar.lark can be found regardless of current working directory
GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), "grammar.lark")


class PatternParser:
    def __init__(self):
        self._parser = Lark.open(
            GRAMMAR_FILE,
            parser="lalr",
            propagate_positions=True,
            maybe_placeholders=False,
        )
        self._transformer = PatternTransformer()

    def parse(self, text: str) -> SequenceNode:
        tree = self._parser.parse(text)
        return self._transformer.transform(tree)


@v_args(inline=True)
class PatternTransformer(Transformer):
    # start -> pattern
    def start(self, pattern):
        return pattern

    # pattern: element+
    def pattern(self, *elements):
        return SequenceNode(list(elements))

    # element: parallel_chain | bracketed | literal
    def element(self, item):
        return item

    # parallel_chain : bracketed (AMP bracketed)+
    @v_args(inline=True)
    def parallel_chain(self, *children):
        nodes = [c for c in children if isinstance(c, PatternNode)]
        return ParallelNode(nodes)

    # parallel_group_inner : LPAR bracketed (AMP bracketed)+ RPAR
    @v_args(inline=True)
    def parallel_group_inner(self, *children):
        nodes = [c for c in children if isinstance(c, PatternNode)]
        return ParallelNode(nodes)

    # bracketed: LBRACK expr RBRACK
    def bracketed(self, _lbrack, expr, _rbrack):
        return expr

    # expr: block_repeat | gap_expr | mod_chain
    def expr(self, node):
        return node

    # gap_tag: COLON TAG
    @v_args(inline=True)
    def gap_tag(self, _colon, tag):
        return tag.value

    # gap_expr: GAP LBRACE INT COMMA INT RBRACE gap_tag?
    @v_args(inline=True)
    def gap_expr(self, _gap, _lbrace, min_num, _comma, max_num, _rbrace, tag=None):
        return GapNode(int(min_num.value), int(max_num.value), tag)

    # modifier: mod_parallel_count | mod_repeat
    @v_args(inline=True)
    def modifier(self, mod_item):
        return mod_item

    # mod_parallel_count : (STAR|HASH) INT parallel_group_inner
    @v_args(inline=True)
    def mod_parallel_count(self, op, num, parallel_block):
        kind = op.value
        cnt = int(num.value)
        # modifier として区別して返す
        return ("PAR", kind, cnt, parallel_block)

    # block_repeat : STAR INT parallel_group_inner
    @v_args(inline=True)
    def block_repeat(self, op, num, parallel_block):
        kind = op.value  # "*" のみが文法で来る
        cnt = int(num.value)
        return ModifierBlockRepeatNode(kind, cnt, parallel_block)

    # mod_repeat : STAR INT | HASH INT
    def mod_repeat(self, op, num):
        kind = op.value
        count = int(num.value)
        return (kind, count)

    # mod_chain: modifier* var_atom
    @v_args(inline=True)
    def mod_chain(self, *items):
        *mods, var_node = items

        for mod in reversed(mods):
            # 旧仕様互換（必要なら）：ModifierRepeatNode
            if isinstance(mod, ModifierRepeatNode):
                mod.head = var_node
                var_node = mod
                continue

            # 並列修飾（*n/#n + 並列ブロック）
            if isinstance(mod, tuple) and len(mod) == 4 and mod[0] == "PAR":
                _, kind, cnt, parallel_block = mod
                var_node = ModifierParallelNode(kind, parallel_block, var_node, count=cnt)
                continue

            # 回数修飾（*n/#n）
            if isinstance(mod, tuple) and len(mod) == 2 and isinstance(mod[1], int):
                kind, cnt = mod
                var_node = ModifierRepeatNode(kind, cnt, var_node)
                continue

            # それ以外（現状は想定しない）
            raise ValueError(f"未知の modifier 形式: {mod}")

        return var_node

    # var_atom: VAR INT pos_tag?
    def var_atom(self, var, idx, pos_tag=None):
        symbol = var.value
        index = int(idx.value)
        tag = pos_tag
        return VariableNode(symbol, index, tag)

    # pos_tag: "-" TAG
    @v_args(inline=True)
    def pos_tag(self, tag):
        return tag.value

    # literal
    def literal(self, tok):
        return LiteralNode([tok.value])
