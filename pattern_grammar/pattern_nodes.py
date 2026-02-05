# pattern_nodes.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Set
from collections import Counter

KIND_TO_LABEL = {"*": "連体修飾", "#": "連用修飾"}


# --------------------------------------------------------------------
#  基底ノード
# --------------------------------------------------------------------
class PatternNode:
    __slots__ = ("children", "seq_id", "parent")

    def __init__(self, children: Optional[List["PatternNode"]] = None):
        self.children: List[PatternNode] = children or []
        self.seq_id: Optional[int] = None
        self.parent: Optional[PatternNode] = None
        # 子ノードに parent ポインタを自動付与
        for c in self.children:
            c.parent = self

    # --------------------------------------------------------------
    #  AST 前順走査
    # --------------------------------------------------------------
    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    # --------------------------------------------------------------
    #  デバッグ出力
    # --------------------------------------------------------------
    def pretty(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}")
        for c in self.children:
            c.pretty(indent + 1)

    def debug(self, indent: int = 0) -> None:
        prefix = "  " * indent
        info = {
            slot: getattr(self, slot, None)
            for slot in getattr(self.__class__, "__slots__", [])
            if slot not in ("children", "parent")
        }
        print(f"{prefix}{self.__class__.__name__} {info}")
        for c in self.children:
            c.debug(indent + 1)

    # --------------------------------------------------------------
    #  API 1: 変数と品詞制約
    # --------------------------------------------------------------
    def get_variable_constraints(self) -> List[Tuple[str, int, Optional[str]]]:
        constraints: List[Tuple[str, int, Optional[str]]] = []
        for node in self.walk():
            if isinstance(node, VariableNode):
                if node.leaf_idx is None:
                    raise ValueError("leaf_idx が未設定の VariableNode があります")
                constraints.append((node.symbol, node.leaf_idx, node.pos_tag))
        return constraints

    # --------------------------------------------------------------
    #  API 2: 依存ラベル要求
    # --------------------------------------------------------------
    def get_dependency_label_requirements(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for node in self.walk():
            label = getattr(node, "dep_label", None)
            if label is None:
                continue
            num = getattr(node, "count", 1)
            counter[label] += num
        return dict(counter)

    # --------------------------------------------------------------
    #  API 3: 必要依存エッジ集合
    # --------------------------------------------------------------
    def get_required_dependency_edges(self) -> Set[Tuple[int, int, str]]:
        var2idx: Dict[Tuple[str, int], int] = {}
        idx = 1
        for node in self.walk():
            if isinstance(node, VariableNode):
                key = (node.symbol, node.index)
                if key not in var2idx:
                    var2idx[key] = idx
                    idx += 1

        edges: Set[Tuple[int, int, str]] = set()
        for node in self.walk():
            if hasattr(node, "dependency_edges"):
                for (fsym, fno), (tsym, tno), label in node.dependency_edges:
                    edges.add((var2idx[(fsym, fno)], var2idx[(tsym, tno)], label))
        return edges

    # --------------------------------------------------------------
    #  API 4: リテラルノード一覧
    # --------------------------------------------------------------
    def get_literal_nodes(self) -> List[Tuple[List[str], Optional[int]]]:
        nodes: List[Tuple[List[str], Optional[int]]] = []
        for node in self.walk():
            if isinstance(node, LiteralNode):
                nodes.append((node.text_tokens, getattr(node, "leaf_idx", None)))
        return nodes


# --------------------------------------------------------------------
#  各種ノードクラス
# --------------------------------------------------------------------
class SequenceNode(PatternNode):
    __slots__ = ("elements",)

    def __init__(self, elements: List[PatternNode]):
        super().__init__(elements)
        self.elements = elements

    def __iter__(self):
        return iter(self.elements)


class VariableNode(PatternNode):
    __slots__ = ("symbol", "index", "pos_tag", "leaf_idx")

    def __init__(self, symbol: str, index: int, pos_tag: Optional[str] = None):
        super().__init__([])
        self.symbol = symbol
        self.index = index
        self.pos_tag = pos_tag
        self.leaf_idx: Optional[int] = None

    def __str__(self):
        tag = f"-{self.pos_tag}" if self.pos_tag else ""
        return f"[{self.symbol}{self.index}{tag}]"


class LiteralNode(PatternNode):
    __slots__ = ("text_tokens", "leaf_idx")

    def __init__(self, text_tokens: List[str]):
        super().__init__([])
        self.text_tokens = text_tokens
        self.leaf_idx: Optional[int] = None

    def __str__(self):
        return "".join(self.text_tokens)




class GapNode(PatternNode):
    __slots__ = ("min_skip", "max_skip", "tag")

    def __init__(self, min_skip: int, max_skip: int, tag: Optional[str] = None):
        super().__init__([])
        if min_skip < 0 or max_skip < 0:
            raise ValueError("GapNode の min_skip / max_skip は 0 以上である必要があります")
        if min_skip > max_skip:
            raise ValueError("GapNode の min_skip は max_skip 以下である必要があります")
        self.min_skip = min_skip
        self.max_skip = max_skip
        # 制約ラベル（任意）。抽出には使わず、マッチ制約として利用する想定。
        self.tag = tag

    def __str__(self):
        tag = f":{self.tag}" if self.tag else ""
        return f"[G{{{self.min_skip},{self.max_skip}}}{tag}]"


class ParallelNode(PatternNode):
    __slots__ = ("options",)

    def __init__(self, options: List[PatternNode]):
        super().__init__(options)
        self.options = options

    # スタブ：実際のマッチングは matcher.py 側で行う
    def matches_any(self, cell):
        return False


class ModifierSingleNode(PatternNode):
    __slots__ = ("kind", "child", "head", "dep_label", "count")

    def __init__(self, kind, child, head, count=1):
        super().__init__([child, head])
        self.kind = kind              # "*" or "#"
        self.child = child
        self.head = head
        self.count = count
        self.dep_label = KIND_TO_LABEL[kind]


class ModifierRepeatNode(PatternNode):
    __slots__ = ("kind", "count", "head", "dep_label")

    def __init__(self, kind: str, count: int, head: PatternNode):
        super().__init__([head])
        self.kind = kind
        self.count = count
        self.head = head
        self.dep_label = KIND_TO_LABEL[kind]


class ModifierBlockRepeatNode(PatternNode):
    __slots__ = ("kind", "count", "block", "head", "dep_label")

    def __init__(self, kind, count, block, head=None):
        super().__init__([block] + ([head] if head else []))
        self.kind = kind
        self.count = count
        self.block = block
        self.head = head
        self.dep_label = KIND_TO_LABEL[kind]


class ModifierParallelNode(PatternNode):
    __slots__ = ("kind", "parallel_block", "head", "dep_label", "count")

    def __init__(self, kind: str, parallel_block: ParallelNode, head: PatternNode, count: int = 1):
        super().__init__([parallel_block, head])
        if count <= 0:
            raise ValueError("ModifierParallelNode の count は 1 以上である必要があります")
        self.kind = kind
        self.parallel_block = parallel_block
        self.head = head
        self.dep_label = KIND_TO_LABEL[kind]
        # 並列修飾も *n/#n の n を保持する（基本は 1）
        self.count = count


# --------------------------------------------------------------------
#  依存エッジノード
# --------------------------------------------------------------------
class DependencyEdgeNode(PatternNode):
    __slots__ = ("from_var", "to_var", "dep_label")

    def __init__(self, from_var: Tuple[str, int],
                 to_var: Tuple[str, int], dep_label: str):
        super().__init__([])
        self.from_var = from_var
        self.to_var = to_var
        self.dep_label = dep_label

    @property
    def dependency_edges(self):
        return [(self.from_var, self.to_var, self.dep_label)]


# --------------------------------------------------------------------
#  Helper functions for AST meta information
# --------------------------------------------------------------------
def extract_literal_strings(ast: PatternNode) -> List[str]:
    """Collect all literal strings in DFS order.

    Parameters
    ----------
    ast: PatternNode
        Root of the AST to traverse.

    Returns
    -------
    list[str]
        A list of literal strings as they appear in depth-first order.
        If the AST contains no ``LiteralNode`` this returns an empty list.
    """

    literals: List[str] = []

    def _visit(node: PatternNode) -> None:
        if isinstance(node, LiteralNode):
            literals.append("".join(node.text_tokens))
        for child in getattr(node, "children", []) or []:
            _visit(child)

    _visit(ast)
    return literals


def count_parallel_variables(ast: PatternNode) -> int:
    """Count VariableNode children directly under any ParallelNode.

    Parameters
    ----------
    ast: PatternNode
        Root of the AST to traverse.

    Returns
    -------
    int
        The number of ``VariableNode`` objects that appear directly under
        ``ParallelNode`` in the AST. If there is no ``ParallelNode`` this
        returns ``0``.
    """

    count = 0

    def _visit(node: PatternNode) -> None:
        nonlocal count
        if isinstance(node, ParallelNode):
            for opt in getattr(node, "options", []) or []:
                if isinstance(opt, VariableNode):
                    count += 1
        for child in getattr(node, "children", []) or []:
            _visit(child)

    _visit(ast)
    return count


# --------------------------------------------------------------------
#  動作デモ（実行すると AST 構造を表示）
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 例: [X1]を[Y2] という最小構成
    x = VariableNode("X", 1, "名詞")
    lit = LiteralNode(["を"])
    y = VariableNode("Y", 2, "動詞")
    dep = DependencyEdgeNode(("X", 1), ("Y", 2), "項-述語")

    ast = SequenceNode([x, lit, y, dep])

    print("=== AST.debug() ===")
    ast.debug()

    print("\n変数制約:", ast.get_variable_constraints())
    print("依存ラベル要求:", ast.get_dependency_label_requirements())
    print("依存エッジ要求:", ast.get_required_dependency_edges())
    print("リテラル一覧:", ast.get_literal_nodes())
