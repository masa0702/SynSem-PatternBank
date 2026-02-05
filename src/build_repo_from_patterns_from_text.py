#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmp/patterns_from_text.jsonl と tmp/patterns_from_manually.jsonl から
- patterns/JA_T2KGB/*.yaml
- index/patterns.jsonl
- index/patterns.index.json
- index/MANIFEST.json
を生成する。

採用条件（dedup）:
1) pattern文字列が完全一致するものは重複として除外
2) pattern_parser.py でパースした AST が同一（= 構造的に同一）なものは重複として除外
   - Xi, Mi, Yi は別物として扱う（ASTが保持している限り、そのまま比較に反映される）

処理はまず tmp/ 内で中間生成物を作り、最後に patterns/ と index/ を更新する。
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -------------------------
# JSONL
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
                yield ln, {"__bad_json__": True, "__error__": str(e), "__raw__": s}


# -------------------------
# ハッシュ
# -------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# AST 正規化（__eq__に依存しない）
# -------------------------
def ast_to_canonical(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [ast_to_canonical(x) for x in obj]
    if isinstance(obj, dict):
        return {k: ast_to_canonical(obj[k]) for k in sorted(obj.keys())}

    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return {"__type__": obj.__class__.__name__, **ast_to_canonical(d)}

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        d = obj.to_dict()
        return {"__type__": obj.__class__.__name__, **ast_to_canonical(d)}

    if hasattr(obj, "__dict__"):
        d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return {"__type__": obj.__class__.__name__, **ast_to_canonical(d)}

    return {"__type__": obj.__class__.__name__, "__repr__": repr(obj)}


def ast_signature(ast_obj: Any) -> str:
    canon = ast_to_canonical(ast_obj)
    s = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------------------------
# index用の軽量特徴（任意）
# -------------------------
_LITERAL_RE = re.compile(r'(?<!\\)"([^"]+)"')
_POS_RE = re.compile(r"-([A-Za-zぁ-んァ-ン一-龥]+)")
_ADJACENT_XY_RE = re.compile(r"\[(X|Y)\d*\]\s*\[(X|Y)\d*\]")
_HIRAGANA_RE = re.compile(r"[ぁ-ん]")


def extract_index_features(pattern_text: str) -> Dict[str, Any]:
    literals = _LITERAL_RE.findall(pattern_text)
    pos_tags = sorted(set(_POS_RE.findall(pattern_text)))
    return {
        "has_literal": bool(_HIRAGANA_RE.search(pattern_text)),
        "literals": literals[:50],
        "pos_constraints": pos_tags[:200],
    }


def has_literal_text(pattern_text: str) -> bool:
    return bool(_HIRAGANA_RE.search(pattern_text))


def ast_has_literal_node(ast_obj: Any) -> bool:
    canon = ast_to_canonical(ast_obj)
    stack = [canon]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            t = cur.get("__type__")
            if isinstance(t, str) and "literal" in t.lower():
                return True
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return False


def has_adjacent_xy(pattern_text: str) -> bool:
    return bool(_ADJACENT_XY_RE.search(pattern_text))


# -------------------------
# pattern文字列の取得（入力JSONLのキーが揺れても耐える）
# -------------------------
CANDIDATE_PATTERN_KEYS = ("pattern", "pattern_str", "pattern_text", "pattern_raw")


def get_pattern_text(rec: Dict[str, Any], pattern_key: Optional[str]) -> Optional[str]:
    if pattern_key:
        v = rec.get(pattern_key)
        return v if isinstance(v, str) and v.strip() else None
    for k in CANDIDATE_PATTERN_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


# -------------------------
# YAML
# -------------------------
def yaml_dump(data: Dict[str, Any]) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML が必要です（docker/requirements.txt に pyyaml を追加してください）")
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        width=120,
        default_flow_style=False,
    )


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../patterns_from_text.jsonl", help="自動生成パターンJSONL")
    ap.add_argument(
        "--input-manual",
        default="../patterns_from_manually.jsonl",
        help="手動作成パターンJSONL（存在しない場合はスキップ）",
    )
    ap.add_argument("--tmp-dir", default="../ver4.0/")
    ap.add_argument("--patterns-dir", default="../ver4.0/patterns/JA_T2KGB")
    ap.add_argument("--index-dir", default="../ver4.0/index")
    ap.add_argument("--pattern-key", default=None, help="入力JSONのパターン文字列キー（指定しないと推測）")
    ap.add_argument("--status", default="draft", choices=("draft", "verified", "deprecated"))
    ap.add_argument("--id-prefix", default="JA_T2KGB_", help="pattern_idのprefix（既定: JA_T2KGB_）")
    ap.add_argument("--id-hash-len", type=int, default=12, help="ASTハッシュから使う長さ（既定: 12）")
    ap.add_argument("--keep-unparsable", action="store_true", help="パース失敗でも採用（AST重複判定は不可）")
    ap.add_argument("--parser-dir", default="../pattern_grammar", help="pattern_parser.py などがあるディレクトリ")
    args = ap.parse_args()

    repo_root = Path(".")
    in_path = repo_root / args.input
    in_manual_path = repo_root / args.input_manual if args.input_manual else None
    tmp_dir = repo_root / args.tmp_dir
    patterns_dir = repo_root / args.patterns_dir
    index_dir = repo_root / args.index_dir
    parser_dir = repo_root / args.parser_dir

    tmp_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # pattern_parser を tmp/pattern_grammar から読み込む
    if not parser_dir.exists():
        raise FileNotFoundError(f"parser-dir が見つかりません: {parser_dir}")

    sys.path.insert(0, str(parser_dir.resolve()))
    try:
        from pattern_parser import PatternParser  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pattern_parser.py を import できません: {e}")

    parser = PatternParser()

    # 中間ファイル（tmp内で処理）
    stage_path = tmp_dir / "_stage_unique.jsonl"
    bad_path = tmp_dir / "bad_records.jsonl"
    report_path = tmp_dir / "build_report.json"

    # 既存のstage/badを潰す（毎回クリーンにしたい人向け）
    if stage_path.exists():
        stage_path.unlink()
    if bad_path.exists():
        bad_path.unlink()

    # 入力リスト（存在しないファイルはスキップ）
    input_paths: List[Path] = []
    if in_path:
        input_paths.append(in_path)
    if in_manual_path and in_manual_path != in_path:
        input_paths.append(in_manual_path)

    missing_inputs: List[str] = []
    existing_inputs: List[Path] = []
    for p in input_paths:
        if p.exists():
            existing_inputs.append(p)
        else:
            missing_inputs.append(str(p))
    if missing_inputs:
        print(f"WARN: missing input files skipped: {', '.join(missing_inputs)}", file=sys.stderr)

    # dedup用
    seen_str = set()
    seen_ast = set()

    stats = {
        "read_lines": 0,
        "read_lines_by_input": {},
        "bad_json": 0,
        "missing_pattern": 0,
        "no_literal": 0,
        "adjacent_xy": 0,
        "dup_string": 0,
        "parse_fail": 0,
        "dup_ast": 0,
        "kept": 0,
        "kept_unparsable": 0,
        "missing_inputs": missing_inputs,
    }

    # 1st pass: streamingで重複除去しつつ stage に落とす
    with stage_path.open("w", encoding="utf-8") as wf_stage, bad_path.open("w", encoding="utf-8") as wf_bad:
        for src_path in existing_inputs:
            src_key = str(src_path)
            stats["read_lines_by_input"][src_key] = 0
            for ln, rec in tqdm(iter_jsonl(src_path), desc=f"dedup+parse:{src_path.name}", unit="line"):
                stats["read_lines"] += 1
                stats["read_lines_by_input"][src_key] += 1

                if rec.get("__bad_json__"):
                    stats["bad_json"] += 1
                    wf_bad.write(
                        json.dumps({"input": src_key, "line": ln, "reason": "bad_json", **rec}, ensure_ascii=False)
                        + "\n"
                    )
                    continue

                ptxt = get_pattern_text(rec, args.pattern_key)
                if not ptxt:
                    stats["missing_pattern"] += 1
                    wf_bad.write(
                        json.dumps(
                            {"input": src_key, "line": ln, "reason": "missing_pattern", "record": rec},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                if has_adjacent_xy(ptxt):
                    stats["adjacent_xy"] += 1
                    wf_bad.write(
                        json.dumps(
                            {"input": src_key, "line": ln, "reason": "adjacent_xy", "pattern": ptxt, "record": rec},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                # 文字列完全一致
                if ptxt in seen_str:
                    stats["dup_string"] += 1
                    continue
                seen_str.add(ptxt)

                # AST パース & 構造重複
                ast_sig = None
                parse_error = None
                try:
                    ast = parser.parse(ptxt)
                    ast_sig = ast_signature(ast)
                    if not (has_literal_text(ptxt) or ast_has_literal_node(ast)):
                        stats["no_literal"] += 1
                        wf_bad.write(
                            json.dumps(
                                {"input": src_key, "line": ln, "reason": "no_literal", "pattern": ptxt, "record": rec},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    if ast_sig in seen_ast:
                        stats["dup_ast"] += 1
                        continue
                    seen_ast.add(ast_sig)
                except Exception as e:
                    parse_error = str(e)
                    stats["parse_fail"] += 1
                    if not has_literal_text(ptxt):
                        stats["no_literal"] += 1
                        wf_bad.write(
                            json.dumps(
                                {"input": src_key, "line": ln, "reason": "no_literal", "pattern": ptxt, "record": rec},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    if not args.keep_unparsable:
                        wf_bad.write(
                            json.dumps(
                                {
                                    "input": src_key,
                                    "line": ln,
                                    "reason": "parse_fail",
                                    "error": parse_error,
                                    "pattern": ptxt,
                                    "record": rec,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                out_min = {
                    "pattern": ptxt,
                    "ast_sig": ast_sig,
                    "parse_error": parse_error,
                    "input_path": src_key,
                    "input_line": ln,
                    "source": rec,  # 元レコードを保持（必要最小限にしたいなら削る）
                }
                wf_stage.write(json.dumps(out_min, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                if ast_sig is None:
                    stats["kept_unparsable"] += 1

    # 2nd pass: stage を読み、pattern_id を確定して patterns/index を生成
    # pattern_id は ASTハッシュ由来で安定化（並び順に依存しない）
    # collision はほぼ起きないが、起きたら末尾に _2, _3 を付けて回避
    id_used = {}

    out_patterns_jsonl = index_dir / "patterns.jsonl"
    out_index_json = index_dir / "patterns.index.json"
    out_manifest_json = index_dir / "MANIFEST.json"

    # 一旦 tmp に書いてから置換（事故りにくい）
    tmp_patterns_jsonl = tmp_dir / "_patterns.jsonl"
    tmp_index_json = tmp_dir / "_patterns.index.json"
    tmp_manifest_json = tmp_dir / "_MANIFEST.json"

    # patterns YAML は大量になりがちなので逐次書く
    index_records: List[Dict[str, Any]] = []

    def make_pattern_id(ast_sig: Optional[str], pattern_text: str) -> str:
        if ast_sig:
            base = f"{args.id_prefix}{ast_sig[:args.id_hash_len]}"
        else:
            # パース不能は文字列ハッシュで（keep_unparsable の時だけ）
            s = hashlib.sha256(pattern_text.encode("utf-8")).hexdigest()
            base = f"{args.id_prefix}{s[:args.id_hash_len]}"
        if base not in id_used:
            id_used[base] = 1
            return base
        id_used[base] += 1
        return f"{base}_{id_used[base]}"

    kept_final = 0

    with stage_path.open("r", encoding="utf-8") as rf, tmp_patterns_jsonl.open("w", encoding="utf-8") as wf_jsonl:
        for line in tqdm(rf, desc="write(patterns+index)", unit="pattern"):
            obj = json.loads(line)
            ptxt = obj["pattern"]
            ast_sig = obj.get("ast_sig")
            pid = make_pattern_id(ast_sig, ptxt)
            source_id = None
            src = obj.get("source")
            if isinstance(src, dict):
                source_id = src.get("id") or src.get("pattern_id")

            # YAML record（必要最小限）
            record = {
                "pattern_id": pid,
                "status": args.status,
                "pattern": ptxt,
            }
            if source_id:
                record["source_id"] = source_id
            if ast_sig:
                record["ast_sig"] = ast_sig
            if obj.get("parse_error"):
                record["parse_error"] = obj["parse_error"]

            # もし入力に “採用後に必要な情報” があるなら、ここで必要分だけ拾う
            # 例: record["meta"] = obj["source"].get("meta")
            # 例: record["extract"] = obj["source"].get("extract")
            # 例: record["slots"] = obj["source"].get("slots")

            out_yaml_path = patterns_dir / f"{pid}.yaml"
            out_yaml_path.write_text(yaml_dump(record), encoding="utf-8")

            # index用
            feats = extract_index_features(ptxt)
            idx_rec = {
                "pattern_id": pid,
                "status": args.status,
                "pattern": ptxt,
                "ast_sig": ast_sig,
                **feats,
            }
            index_records.append(
                {
                    "pattern_id": pid,
                    "status": args.status,
                    "ast_sig": ast_sig,
                    "has_literal": feats["has_literal"],
                    "pos_constraints": feats["pos_constraints"],
                    # literals は重いので index.json 側には入れず、必要なら patterns.jsonl 側で持たせる
                }
            )

            wf_jsonl.write(json.dumps(idx_rec, ensure_ascii=False) + "\n")
            kept_final += 1

    # patterns.index.json（軽量）
    tmp_index_json.write_text(
        json.dumps(
            {
                "count": kept_final,
                "patterns": index_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # MANIFEST
    manifest_inputs = []
    for p in input_paths:
        manifest_inputs.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() else None,
            }
        )

    manifest = {
        "inputs": manifest_inputs,
        "parser": {
            "dir": str(parser_dir),
            "grammar_sha256": sha256_file(parser_dir / "grammar.lark") if (parser_dir / "grammar.lark").exists() else None,
        },
        "output": {
            "patterns_dir": str(patterns_dir),
            "index_dir": str(index_dir),
            "patterns_count": kept_final,
        },
        "stats": stats,
        "filters": {
            "require_literal": True,
            "reject_adjacent_xy": True,
            "adjacent_xy_rule": "X/Y要素が &・リテラル・[G] なしで横並び（[X*][Y*] 等の直接連結）",
        },
        "dedup_policy": {
            "string_exact": True,
            "structural_ast": True,
            "ast_sig": "sha256(canonicalized_ast)",
        },
    }
    tmp_manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # レポート（tmp）
    report_path.write_text(json.dumps({"stats": stats, "kept_final": kept_final}, ensure_ascii=False, indent=2), encoding="utf-8")

    # index/ を更新（置換）
    os.replace(tmp_patterns_jsonl, out_patterns_jsonl)
    os.replace(tmp_index_json, out_index_json)
    os.replace(tmp_manifest_json, out_manifest_json)

    print("DONE")
    print(f"- patterns written: {kept_final}")
    print(f"- index: {out_patterns_jsonl}, {out_index_json}, {out_manifest_json}")
    print(f"- tmp reports: {report_path}, {bad_path}")


if __name__ == "__main__":
    main()
