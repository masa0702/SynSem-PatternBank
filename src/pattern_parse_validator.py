# /workspace/src/pattern_parse_validator.py
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParseError:
    index: int
    pattern: str
    message: str


def _load_pattern_parser_class(parser_path: str):
    """
    pattern_parser.py をファイルパスから動的にロードして PatternParser クラスを返す。

    重要:
    - pattern_parser.py が同一ディレクトリ内の pattern_nodes 等を import できるように、
      parser のディレクトリを sys.path に追加してからロードする。
    """
    import sys
    import importlib.util
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent

    p = Path(parser_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"pattern_parser.py が見つかりません: {p}")

    parser_dir = str(p.parent)

    # pattern_nodes など同ディレクトリの import を通すため、先頭に追加（重複は避ける）
    if parser_dir not in sys.path:
        sys.path.insert(0, parser_dir)

    # ここで pattern_parser.py をロード
    spec = importlib.util.spec_from_file_location("pattern_parser_dynamic", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"pattern_parser.py の import spec 作成に失敗しました: {p}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "PatternParser"):
        raise AttributeError(f"PatternParser が {p} に定義されていません。")

    return getattr(mod, "PatternParser")


class PatternParseValidator:
    """
    生成された PatternRepoOutput（patterns配列）を PatternParser で検証する。
    """

    def __init__(self, parser_path: str):
        PatternParser = _load_pattern_parser_class(parser_path)
        self._parser = PatternParser()

    def validate_pattern(self, pattern: str) -> Optional[str]:
        """
        成功: None
        失敗: エラーメッセージ(str)
        """
        try:
            self._parser.parse(pattern)
            return None
        except Exception as e:
            return str(e)

    def validate_output(self, obj: Dict[str, Any]) -> Tuple[bool, List[ParseError]]:
        errors: List[ParseError] = []
        patterns = obj.get("patterns", [])
        for i, p in enumerate(patterns):
            pat = p.get("pattern", "")
            msg = self.validate_pattern(pat)
            if msg is not None:
                errors.append(ParseError(index=i, pattern=pat, message=msg))
        return (len(errors) == 0), errors
