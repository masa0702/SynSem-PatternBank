# test_pattern_parse.py
# 目的: pattern_parser.py / grammar.lark のパース可否を、最新の仕様に照らして検証する。
#
# 使い方（例）:
#   python test_pattern_parse.py
#   python test_pattern_parse.py --visualize --out_dir test_asts
#
# 前提（最新版）:
# - grammar.lark: 
#     * []内の&は禁止
#     * 並列修飾は必ず *n/#n を伴う（例: [*1([M0]&[M1])X0]）
#     * stand-alone block は [*1([X0]&[X1])] のみ許可（# や単一要素は不可）
#     * トップレベル並列は [X0]&[X1] の形式（括弧不要）
# - pattern_nodes.py: GapNode / ModifierParallelNode(count対応) を含む
# - pattern_parser_v3.py を利用する場合は pattern_parser.py に置換するか、
#   本スクリプトの自動importに任せる

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# ------------------------------
# パーサの自動import（v3→v2→v1）
# ------------------------------
def _import_parser():
    last_err = None
    for mod_name in ("pattern_parser", "pattern_parser_v3", "pattern_parser_v2"):
        try:
            mod = __import__(mod_name, fromlist=["PatternParser"])
            return mod.PatternParser
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "PatternParser の import に失敗しました。"
        " pattern_parser.py / pattern_parser_v3.py の配置と依存関係を確認してください。\n"
        f"最後の例外: {last_err}"
    )

PatternParser = _import_parser()

# visualize は任意（Graphvizが無い環境でもテストは回す）
def _try_import_visualizer():
    try:
        from ast_visualizer import visualize_ast
        return visualize_ast
    except Exception:
        return None

visualize_ast = _try_import_visualizer()

# ------------------------------
# テストケース定義
# ------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    pattern: str
    should_parse: bool = True
    note: str = ""

def build_cases() -> List[Case]:
    cases: List[Case] = []

    # --- 基本（変数・リテラル） ---
    cases += [
        Case("basic_var_X", "[X0]"),
        Case("basic_var_Y", "[Y0]"),
        Case("basic_var_M", "[M0]"),
        Case("basic_literal_mix", "は[X0]が[Y0]する"),
    ]

    # --- 品詞タグ（pos_tag） ---
    cases += [
        Case("pos_tag_simple", "[X0-名詞]"),
        Case("pos_tag_hyphenated", "[X1-名詞-固有名詞]"),
        Case("pos_tag_plus", "[Y0-サ変+する]"),
    ]

    # --- 回数修飾（*n / #n） ---
    cases += [
        Case("mod_repeat_star", "[*1X0]"),
        Case("mod_repeat_hash", "[#2X0]"),
        Case("mod_repeat_chain", "[*1#2X0-名詞]"),
    ]

    # --- トップレベル並列（括弧なし） ---
    cases += [
        Case("parallel_chain_top_level", "[X0]&[X1]"),
        Case("parallel_chain_top_level_mixed", "[X0-名詞]&[M1-名詞]"),
    ]

    # --- [] 内の & は禁止 ---
    cases += [
        Case("parallel_expr_in_bracket", "[X0&X1]", should_parse=False),
        Case("parallel_expr_nested_atom", "[[X0]&[X1]]", should_parse=False),
    ]

    # --- 並列修飾（必ず *n/#n） ---
    cases += [
        Case("mod_parallel_count_star", "[*1([M0]&[M1])X0]"),
        Case("mod_parallel_count_hash", "[#2([M0]&[M1])X0]"),
        Case("neg_mod_parallel_no_count_star", "[*([M0]&[M1])X0]", should_parse=False,
             note="並列修飾は必ず *n/#n を伴う"),
        Case("neg_mod_parallel_no_count_hash", "[#([M0]&[M1])X0]", should_parse=False,
             note="並列修飾は必ず *n/#n を伴う"),
    ]

    # --- stand-alone block（許可: [*1([X0]&[X1])] のみ） ---
    cases += [
        Case("block_repeat_parallel_expr", "[*1([X0]&[X1])]"),
        Case("neg_block_repeat_hash", "[#2([X0]&[X1])]", should_parse=True,
             note="stand-alone block は */# で許可"),
        Case("neg_block_repeat_single_atom", "[#2(X0)]", should_parse=False,
             note="stand-alone block の単一要素は不許可"),
        Case("neg_block_repeat_single_atom_star", "[*1(X0)]", should_parse=False,
             note="stand-alone block の単一要素は不許可"),
    ]

    # --- ギャップ（G{m,n} と任意タグ） ---
    cases += [
        Case("gap_basic_0_2", "[G{0,2}]"),
        Case("gap_basic_1_2", "[G{1,2}]"),
        Case("gap_with_tag", "[G{1,2}:TIME]"),
    ]

    # --- ギャップを含む複合（仕様: 修飾無し並列は括弧を付けない） ---
    cases += [
        Case(
            "gap_with_parallel_and_relation",
            "[X0]は[G{1,2}][X2]&[X3]が[Y0]として"
        ),
        Case(
            "gap_with_particles",
            "[X0]は[G{0,2}][X1]が[Y0]した"
        ),
        Case(
            "modifier_then_gap",
            "[*1X0]は[G{0,2}][X1]が[Y0]する"
        ),
    ]

    # --- ネガティブ（パース失敗を期待） ---
    cases += [
        Case("neg_gap_missing_comma", "[G{1 2}]", should_parse=False),
        Case("neg_gap_reversed_range", "[G{2,1}]", should_parse=False),
        Case("neg_unclosed_bracket", "[X0", should_parse=False),
        Case("neg_gap_no_upper", "[G{1,}]", should_parse=False),
        Case("neg_parenthesized_top_parallel", "([X0]&[X1])", should_parse=False,
             note="トップレベル並列は括弧を要求しない（最小化）。括弧付きは仕様外。"),
    ]

    return cases

# ------------------------------
# 実行本体
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", action="store_true", help="AST可視化（Graphvizがある場合のみ有効）")
    ap.add_argument("--out_dir", default="test_asts", help="可視化の出力ディレクトリ")
    args = ap.parse_args()

    parser = PatternParser()
    cases = build_cases()

    ok = 0
    ng = 0
    unexpected: List[Tuple[Case, str]] = []

    if args.visualize and visualize_ast is None:
        print("注意: ast_visualizer の import に失敗したため、可視化はスキップします。")
        args.visualize = False

    if args.visualize:
        os.makedirs(args.out_dir, exist_ok=True)

    for c in cases:
        try:
            ast = parser.parse(c.pattern)
            if not c.should_parse:
                ng += 1
                unexpected.append((c, "本来は失敗すべきだが成功した（文法が仕様より緩い可能性）"))
                print(f"[UNEXPECTED-OK] {c.name}: {c.pattern}")
                if c.note:
                    print(f"  note: {c.note}")
                continue

            ok += 1
            print(f"[OK] {c.name}: {c.pattern}")
            if c.note:
                print(f"  note: {c.note}")

            try:
                ast.debug()
            except Exception:
                pass

            if args.visualize:
                safe_name = "".join(ch if ch.isalnum() else "_" for ch in c.name)
                try:
                    visualize_ast(ast, out_filename=safe_name, output_dir=args.out_dir, view=False)
                except Exception as e:
                    print(f"  可視化スキップ（失敗）: {e}")

        except Exception as e:
            if c.should_parse:
                ng += 1
                unexpected.append((c, f"本来は成功すべきだが失敗: {e}（文法が仕様より厳しい可能性）"))
                print(f"[NG] {c.name}: {c.pattern}")
                print(f"  → {e}")
                if c.note:
                    print(f"  note: {c.note}")
            else:
                ok += 1
                print(f"[OK-EXPECTED-FAIL] {c.name}: {c.pattern}")
                if c.note:
                    print(f"  note: {c.note}")
                print(f"  （期待通り失敗）→ {e}")

    print("\n====================")
    print(f"総数: {len(cases)} / OK: {ok} / NG: {ng}")
    if unexpected:
        print("\n想定と異なる結果（仕様と文法の差分候補）:")
        for c, msg in unexpected:
            print(f"- {c.name}: {msg}\n  pattern={c.pattern}")
            if c.note:
                print(f"  note={c.note}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
