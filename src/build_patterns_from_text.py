# build_patterns_from_text.py
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


# -------------------------
# ログ（日本語）
# -------------------------
logger = logging.getLogger("build_patterns_from_text")


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# -------------------------
# JSONL I/O
# -------------------------
def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} の {ln} 行目が不正なJSONです: {e}") from e


def append_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# 探索
# -------------------------
def find_files_recursive(root: Path, names: List[str]) -> List[Path]:
    """
    root 配下を再帰的に走査して、ファイル名が names のいずれかに一致するものを返す。
    """
    hits: List[Path] = []
    name_set = set(names)
    for p in root.rglob("*"):
        if p.is_file() and p.name in name_set:
            hits.append(p)
    hits.sort()
    return hits


@dataclass(frozen=True)
class BuildConfig:
    require_keys: List[str]
    add_source: bool


def validate_record(obj: Dict[str, Any], require_keys: List[str]) -> Tuple[bool, str]:
    """
    最低限、要求キーが存在するかだけチェックする。
    """
    if not isinstance(obj, dict):
        return False, "レコードがdictではありません"
    for k in require_keys:
        if k not in obj:
            return False, f"必須キー欠落: {k}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ディレクトリ内の pass_pair(s).jsonl を再帰探索し patterns_from_text.jsonl を作成します",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 入力
    ap.add_argument("--root_dir", type=str, default="../pattern_candidate", help="探索ルートディレクトリ")
    ap.add_argument(
        "--pass_names",
        type=str,
        default="pass_pairs.jsonl,pass_pair.jsonl,pass_pairs_all.jsonl",
        help="集約対象ファイル名（カンマ区切り）",
    )

    # 出力
    ap.add_argument("--out", type=str, default="patterns_from_text.jsonl", help="完成系の出力JSONL")
    ap.add_argument("--bad_out", type=str, default="bad_records.jsonl", help="形式不正行の退避JSONL")
    ap.add_argument("--report", type=str, default="build_report.json", help="集約レポートJSON")

    # 振る舞い
    ap.add_argument(
        "--require_keys",
        type=str,
        default="sent_ja,pattern",
        help="最低限含まれていて欲しいキー（カンマ区切り）",
    )
    ap.add_argument(
        "--add_source",
        action="store_true",
        default=False,
        help="各行に __source（元ファイル/行番号）を付与する（後段の重複除去で邪魔ならOFF推奨）",
    )

    # ログ
    ap.add_argument("--log_level", type=str, default="INFO", help="ログレベル")

    args = ap.parse_args()
    setup_logger(args.log_level)

    root = Path(args.root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"root_dir がディレクトリではありません: {root}")

    pass_names = [s.strip() for s in args.pass_names.split(",") if s.strip()]
    require_keys = [s.strip() for s in args.require_keys.split(",") if s.strip()]
    cfg = BuildConfig(require_keys=require_keys, add_source=args.add_source)

    out_path = Path(args.out)
    bad_out_path = Path(args.bad_out)
    report_path = Path(args.report)

    # 走査
    logger.info("探索開始: root=%s", str(root))
    pass_files = find_files_recursive(root, pass_names)
    logger.info("対象 pass ファイル数: %d", len(pass_files))

    if not pass_files:
        logger.warning("対象ファイルが見つかりませんでした。pass_names=%s", pass_names)

    # 集約
    total_in_files = 0
    total_in_records = 0
    total_out_records = 0
    total_bad_records = 0
    per_file_counts: Dict[str, Dict[str, int]] = {}
    bad_reasons = Counter()

    with out_path.open("w", encoding="utf-8") as fp_out, bad_out_path.open("w", encoding="utf-8") as fp_bad:
        for fpath in tqdm(pass_files, desc="集約（ファイル）", unit="file"):
            total_in_files += 1

            ok_cnt = 0
            bad_cnt = 0

            for ln, obj in tqdm(iter_jsonl(fpath), desc=f"読込: {fpath.name}", unit="rec", leave=False):
                total_in_records += 1

                ok, reason = validate_record(obj, cfg.require_keys)
                if not ok:
                    bad_cnt += 1
                    total_bad_records += 1
                    bad_reasons[reason] += 1
                    rec2 = {
                        "__error": {"reason": reason, "source_file": str(fpath), "source_lineno": ln},
                        "raw": obj,
                    }
                    append_jsonl_line(fp_bad, rec2)
                    continue

                if cfg.add_source:
                    obj = dict(obj)
                    obj["__source"] = {"file": str(fpath), "lineno": ln}

                append_jsonl_line(fp_out, obj)
                ok_cnt += 1
                total_out_records += 1

            per_file_counts[str(fpath)] = {"ok": ok_cnt, "bad": bad_cnt}

    report = {
        "root_dir": str(root),
        "pass_names": pass_names,
        "out": str(out_path),
        "bad_out": str(bad_out_path),
        "require_keys": require_keys,
        "add_source": cfg.add_source,
        "input_files": total_in_files,
        "input_records": total_in_records,
        "output_records": total_out_records,
        "bad_records": total_bad_records,
        "bad_reasons": dict(bad_reasons),
        "per_file_counts": per_file_counts,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("完了: out=%s（%d行） bad=%s（%d行）", str(out_path), total_out_records, str(bad_out_path), total_bad_records)
    logger.info("レポート: %s", str(report_path))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
