# eval_generated_patterns_dir.py
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

# 提示された bunsetu.py を利用（生成時と同じ文節分割）
from bunsetu import BunsetsuSegmenter


# -------------------------
# ログ（日本語）
# -------------------------
logger = logging.getLogger("pattern_eval")


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# -------------------------
# JSONL I/O（1行=1JSON）
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
# 文節チェック用
# -------------------------
PUNCT_ONLY_RE = re.compile(r"^[\s、。,.!?！？「」『』（）()\[\]【】]+$")


@dataclass(frozen=True)
class EvalConfig:
    """評価設定"""
    ignore_punct_only_bunsetsu: bool = True


def list_jsonl_files(input_dir: Path, pattern: str, recursive: bool, exclude_names: List[str]) -> List[Path]:
    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))

    exclude_set = set(exclude_names)
    out: List[Path] = []
    for p in files:
        if p.is_file() and p.name not in exclude_set:
            out.append(p)
    return out


def get_bunsetsu_texts(segmenter: BunsetsuSegmenter, sentence: str, cfg: EvalConfig) -> List[str]:
    """
    bunsetu.py の BunsetsuSegmenter.segment(sentence) の返り値（リスト）の先頭要素を文節文字列として扱う。
    """
    bunsetsu_list = segmenter.segment(sentence)
    texts: List[str] = []
    for b in bunsetsu_list:
        if not b or not isinstance(b, list):
            continue
        t = str(b[0])
        if not t.strip():
            continue
        if cfg.ignore_punct_only_bunsetsu and PUNCT_ONLY_RE.match(t):
            continue
        texts.append(t)
    return texts


# -------------------------
# 評価本体
# -------------------------
def eval_record(rec: Dict[str, Any], segmenter: BunsetsuSegmenter, cfg: EvalConfig) -> Tuple[bool, List[str]]:
    """
    1行レコードを評価して (pass, 理由リスト) を返す。

    pass 条件（最新版）:
      1) error なし
      2) 必須フィールド（sent_ja / final_triples_indexed / patterns）が存在し非空
      3) triple idx が全て int で、pattern.covers が存在する triple idx のみ参照
      4) covers の union が triple idx 集合と完全一致（全 triple をカバー）
      5) triple>=2 かつ pattern==1 の場合は '&' を含む（含まなければ不採用）
      6) bunsetu.py で文節分割し、文節文字列が pattern にそのまま含まれていたら不採用
    """
    reasons: List[str] = []

    # 1: errorになっていないか
    if rec.get("error"):
        reasons.append("エラーが記録されています")

    sent = rec.get("sent_ja")
    if not isinstance(sent, str) or not sent.strip():
        reasons.append("sent_ja が空、または欠落しています")

    triples = rec.get("final_triples_indexed")
    if not isinstance(triples, list) or len(triples) == 0:
        reasons.append("final_triples_indexed が空、または欠落しています")

    patterns = rec.get("patterns")
    if not isinstance(patterns, list) or len(patterns) == 0:
        reasons.append("patterns が空、または欠落しています")

    if reasons:
        return False, reasons

    # triple idx 収集
    triple_idxs: List[int] = []
    for t in triples:
        if not isinstance(t, dict) or "idx" not in t or not isinstance(t["idx"], int):
            reasons.append("final_triples_indexed の形式が不正です（idx がありません）")
            continue
        triple_idxs.append(t["idx"])

    if reasons:
        return False, reasons

    triple_idx_set = set(triple_idxs)

    # covers の妥当性 + 全カバー（union一致）
    covered_all: List[int] = []
    for p in patterns:
        if not isinstance(p, dict):
            reasons.append("patterns 内に不正な要素があります（dict ではありません）")
            continue

        cov = p.get("covers")
        pat = p.get("pattern")

        if not isinstance(cov, list) or any(not isinstance(x, int) for x in cov):
            reasons.append("covers の形式が不正です（int のリストではありません）")
            continue
        if not isinstance(pat, str) or not pat.strip():
            reasons.append("pattern が空、または欠落しています")
            continue

        for x in cov:
            if x not in triple_idx_set:
                reasons.append("covers が存在しない triple idx を参照しています")
                break
            covered_all.append(x)

    if reasons:
        return False, sorted(set(reasons))

    if set(covered_all) != triple_idx_set:
        reasons.append("covers が全 triple をカバーしていません")

    # triple>=2 かつ pattern==1 の場合は '&' 必須
    if len(triples) >= 2 and len(patterns) == 1:
        p0 = patterns[0].get("pattern", "")
        if not (isinstance(p0, str) and ("&" in p0)):
            reasons.append("複数tripleでパターンが1つですが、'&' が含まれていません（不採用）")

    # 文節対応（bunsetu.py）
    try:
        bunsetsu_texts = get_bunsetsu_texts(segmenter, sent, cfg)
    except Exception as e:
        reasons.append(f"文節分割（bunsetu.py）に失敗しました: {e}")
        return False, sorted(set(reasons))

    # 文節が pattern にそのまま含まれていたら不採用
    for p in patterns:
        pat = p.get("pattern", "")
        if not isinstance(pat, str):
            reasons.append("pattern が文字列ではありません")
            continue
        for bt in bunsetsu_texts:
            if bt and bt in pat:
                reasons.append("パターンに文節文字列がそのまま含まれています（不採用）")
                break
        if "パターンに文節文字列がそのまま含まれています（不採用）" in reasons:
            break

    ok = len(reasons) == 0
    return ok, sorted(set(reasons))


def flatten_pass_pattern_sentence_pairs(rec: Dict[str, Any], source_file: str, lineno: int) -> List[Dict[str, Any]]:
    """
    pass レコードを「パターン + 文」単位で出力する（1行=1パターン）。
    covers が複数でも許容するため、ここでは分割しない。
    """
    sent = rec["sent_ja"]
    triples = {t["idx"]: t for t in rec["final_triples_indexed"]}
    out: List[Dict[str, Any]] = []

    for p in rec["patterns"]:
        cov = p.get("covers", [])
        cov_triples = []
        if isinstance(cov, list):
            for idx in cov:
                if idx in triples:
                    t = triples[idx]
                    cov_triples.append({
                        "idx": idx,
                        "sub": t.get("sub"),
                        "rel": t.get("rel"),
                        "obj": t.get("obj"),
                    })

        out.append({
            "source_file": source_file,
            "source_lineno": lineno,
            "id": rec.get("id"),
            "sent_ja": sent,
            "pattern": p.get("pattern"),
            "covers": cov,
            "triples_covered": cov_triples,
        })

    return out


# -------------------------
# no_pass の id からオリジナルを回収
# -------------------------
def export_no_pass_origin(
    no_pass_ids: Set[str],
    origin_dir: Path,
    origin_glob: str,
    origin_recursive: bool,
    out_path: Path,
) -> Dict[str, Any]:
    """
    ../data/ 配下の ont_*_ground_truth.jsonl を走査し、id が一致するレコードを no_pass_origin.jsonl に保存する。
    """
    if not no_pass_ids:
        out_path.write_text("", encoding="utf-8")
        return {"探索対象id数": 0, "発見数": 0, "未発見数": 0}

    origin_files = list_jsonl_files(
        input_dir=origin_dir,
        pattern=origin_glob,
        recursive=origin_recursive,
        exclude_names=[],
    )

    found = 0
    seen_ids: Set[str] = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for fpath in tqdm(origin_files, desc="オリジナル探索", unit="file"):
            for ln, rec in tqdm(iter_jsonl(fpath), desc=f"探索中: {fpath.name}", unit="rec", leave=False):
                rid = rec.get("id")
                if not isinstance(rid, str):
                    continue
                if rid in no_pass_ids and rid not in seen_ids:
                    # 元データに最小限の追跡情報だけ付与（邪魔なら削除してOK）
                    rec2 = dict(rec)
                    rec2["__origin"] = {"source_file": fpath.name, "source_lineno": ln}
                    append_jsonl_line(fp, rec2)
                    seen_ids.add(rid)
                    found += 1

    missing = len(no_pass_ids) - len(seen_ids)
    return {"探索対象id数": len(no_pass_ids), "発見数": found, "未発見数": missing}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="生成パターンJSONLをまとめて評価し、pass/no-passと no_pass_origin を出力します（ログ・コメント日本語）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # すべて default 設定
    ap.add_argument("--input_dir", type=str, default="../output/ver_3", help="入力JSONLが入ったディレクトリ")
    ap.add_argument("--glob", type=str, default="*.jsonl", help="入力ファイルのglobパターン")
    ap.add_argument("--recursive", action="store_true", default=False, help="サブディレクトリも再帰的に探索する")
    ap.add_argument("--out_dir", type=str, default="../eval_pattern/eval_pattern_v3", help="出力先ディレクトリ（空なら自動で timestamp 生成）")
    ap.add_argument(
        "--exclude",
        type=str,
        default="pass_pairs.jsonl,no_pass.jsonl,report.json,no_pass_origin.jsonl",
        help="処理対象から除外するファイル名（カンマ区切り）",
    )
    ap.add_argument("--log_level", type=str, default="INFO", help="ログレベル（INFO/DEBUG/WARNING/ERROR）")

    # 追加: オリジナル探索の設定（defaultあり）
    ap.add_argument("--origin_dir", type=str, default="../data", help="オリジナル（ground_truth）JSONLが入ったディレクトリ")
    ap.add_argument("--origin_glob", type=str, default="ont_*_ground_truth.jsonl", help="オリジナルJSONLのglob")
    ap.add_argument("--origin_recursive", action="store_true", default=False, help="オリジナル探索を再帰的に行う")
    ap.add_argument("--origin_out_name", type=str, default="../data/no_pass_v3/no_pass_origin.jsonl", help="オリジナル回収結果の出力ファイル名")

    args = ap.parse_args()
    setup_logger(args.log_level)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")

    # 出力先
    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("pattern_eval_out") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_path = out_dir / "pass_pairs.jsonl"
    nopass_path = out_dir / "no_pass.jsonl"
    report_path = out_dir / "report.json"
    origin_out_path = out_dir / args.origin_out_name

    exclude_names = [s.strip() for s in args.exclude.split(",") if s.strip()]
    files = list_jsonl_files(input_dir, args.glob, args.recursive, exclude_names)

    logger.info("評価を開始します。入力ディレクトリ=%s 対象ファイル数=%d 出力=%s", str(input_dir), len(files), str(out_dir))
    if not files:
        logger.warning("対象ファイルが0件です。glob=%s recursive=%s", args.glob, args.recursive)

    # 文節分割器は1回だけ初期化（重いので）
    segmenter = BunsetsuSegmenter()
    cfg = EvalConfig()

    # 集計
    global_fail_counter = Counter()
    total_records = 0
    pass_records = 0
    pass_pairs = 0  # 「パターン+文」行数

    # 追加: no_pass の id 集合
    no_pass_ids: Set[str] = set()

    with pass_path.open("w", encoding="utf-8") as fp_pass, nopass_path.open("w", encoding="utf-8") as fp_nopass:
        for fpath in tqdm(files, desc="ファイル処理", unit="file"):
            for ln, rec in tqdm(iter_jsonl(fpath), desc=f"評価中: {fpath.name}", unit="rec", leave=False):
                total_records += 1

                ok, reasons = eval_record(rec, segmenter, cfg)
                if ok:
                    pass_records += 1
                    pairs = flatten_pass_pattern_sentence_pairs(rec, fpath.name, ln)
                    for obj in pairs:
                        append_jsonl_line(fp_pass, obj)
                    pass_pairs += len(pairs)
                else:
                    rid = rec.get("id")
                    if isinstance(rid, str):
                        no_pass_ids.add(rid)

                    for r in reasons:
                        global_fail_counter[r] += 1

                    rec2 = dict(rec)
                    rec2["__eval"] = {"pass": False, "reasons": reasons, "lineno": ln, "source_file": fpath.name}
                    append_jsonl_line(fp_nopass, rec2)

    # 追加: no_pass の id を使って ../data から元データを回収
    origin_stats = export_no_pass_origin(
        no_pass_ids=no_pass_ids,
        origin_dir=Path(args.origin_dir),
        origin_glob=args.origin_glob,
        origin_recursive=args.origin_recursive,
        out_path=origin_out_path,
    )

    report = {
        "input_dir": str(input_dir),
        "glob": args.glob,
        "recursive": args.recursive,
        "output_dir": str(out_dir),
        "pass_pairs_jsonl": str(pass_path),
        "no_pass_jsonl": str(nopass_path),
        "no_pass_origin_jsonl": str(origin_out_path),
        "総レコード数": total_records,
        "passレコード数": pass_records,
        "pass（パターン+文）行数": pass_pairs,
        "no_passレコード数": total_records - pass_records,
        "no_pass理由集計": dict(global_fail_counter),
        "no_pass_origin回収": origin_stats,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("評価が完了しました。総=%d pass=%d pass行数=%d no_pass=%d",
                total_records, pass_records, pass_pairs, total_records - pass_records)
    logger.info("no_pass_origin 回収: 対象id=%d 発見=%d 未発見=%d",
                origin_stats.get("探索対象id数", 0), origin_stats.get("発見数", 0), origin_stats.get("未発見数", 0))
    logger.info("出力: %s", str(out_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
