# text2pattern_regen_until_pass.py
# -*- coding: utf-8 -*-

"""
テキストからパターン生成 → 評価 → no_pass のみ再生成 → … を繰り返し、
no_pass が 0 になるまで（または上限回数まで）回すオーケストレータ。

依存:
- text2pattern_batch_multi_gpus.py（生成）
- eval_generated_patterns_dir.py（評価 + no_pass_origin 回収）
- bunsetu.py（評価で使用）
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

# 生成側（あなたの現行スクリプト）
from text2pattern_batch_multi_gpus import (
    process_one_file_parallel,
    process_one_file_single,
    parse_ollama_urls,
)

# 評価側（あなたの現行スクリプト）
from eval_generated_patterns_dir import (
    iter_jsonl as iter_jsonl_eval,
    append_jsonl_line as append_jsonl_line_eval,
    list_jsonl_files as list_jsonl_files_eval,
    eval_record,
    flatten_pass_pattern_sentence_pairs,
    export_no_pass_origin,
    EvalConfig,
)

from bunsetu import BunsetsuSegmenter


# -------------------------
# ログ（日本語）
# -------------------------
logger = logging.getLogger("t2p_regen")


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# -------------------------
# ユーティリティ
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_inputs(in_dir: Path, glob_pat: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(in_dir.rglob(glob_pat))
    return sorted(in_dir.glob(glob_pat))


# -------------------------
# 生成（複数ファイル or 単一ファイル）
# -------------------------
def run_generation_for_files(
    in_files: List[Path],
    out_dir: Path,
    schema_path: Path,
    disable_parallel: bool,
    ollama_base_url: str,
    ollama_urls: List[str],
    model: str,
    endpoint: str,
    max_tokens: int,
    num_ctx: int | None,
    think: str,
    timeout: int,
    max_retries: int,
    max_workers: int,
    tmp_root: Path,
    keep_tmp: bool,
    progress_chunk: int,
) -> List[Path]:
    """
    入力ファイルごとに t2p_{name}.jsonl を out_dir に生成して返す。
    """
    ensure_dir(out_dir)
    ensure_dir(tmp_root)

    out_files: List[Path] = []
    for path in tqdm(in_files, desc="パターン生成（ファイル）", unit="file"):
        out_path = out_dir / f"t2p_{path.name}"
        logger.info("生成開始: %s -> %s", path.name, out_path.name)

        if disable_parallel or len(ollama_urls) <= 1:
            process_one_file_single(
                path,
                out_path,
                schema_path=schema_path,
                ollama_base_url=ollama_base_url,
                model=model,
                endpoint=endpoint,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
                think=think,
                timeout=timeout,
                max_retries=max_retries,
                max_workers=max_workers,
            )
        else:
            process_one_file_parallel(
                path,
                out_path,
                schema_path=schema_path,
                ollama_urls=ollama_urls,
                model=model,
                endpoint=endpoint,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
                think=think,
                timeout=timeout,
                max_retries=max_retries,
                max_workers=max_workers,
                tmp_root=tmp_root,
                keep_tmp=keep_tmp,
                progress_chunk=progress_chunk,
            )

        logger.info("生成完了: %s", out_path.name)
        out_files.append(out_path)

    return out_files


# -------------------------
# 評価（生成結果dir → pass/no_pass/no_pass_origin）
# -------------------------
def run_evaluation_dir(
    gen_dir: Path,
    gen_glob: str,
    out_dir: Path,
    origin_dir: Path,
    origin_glob: str,
    origin_recursive: bool,
    segmenter: BunsetsuSegmenter,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    gen_dir 内の生成結果 JSONL を評価して、以下を out_dir に作る。
      - pass_pairs.jsonl（1行=パターン+文）
      - no_pass.jsonl（元レコード + __eval）
      - no_pass_origin.jsonl（../data からオリジナル回収）
    戻り値には no_pass_ids などを含める。
    """
    ensure_dir(out_dir)

    pass_path = out_dir / "pass_pairs.jsonl"
    nopass_path = out_dir / "no_pass.jsonl"
    origin_path = out_dir / "no_pass_origin.jsonl"
    report_path = out_dir / "eval_report.json"

    # 評価対象ファイル
    files = list_jsonl_files_eval(gen_dir, gen_glob, recursive=False, exclude_names=[])
    logger.info("評価対象: %d ファイル（%s/%s）", len(files), str(gen_dir), gen_glob)

    global_fail_counter: Dict[str, int] = {}
    total_records = 0
    pass_records = 0
    pass_lines = 0
    no_pass_ids: Set[str] = set()

    with pass_path.open("w", encoding="utf-8") as fp_pass, nopass_path.open("w", encoding="utf-8") as fp_nopass:
        for fpath in tqdm(files, desc="評価（ファイル）", unit="file"):
            for ln, rec in tqdm(iter_jsonl_eval(fpath), desc=f"評価中: {fpath.name}", unit="rec", leave=False):
                total_records += 1
                ok, reasons = eval_record(rec, segmenter, cfg)

                if ok:
                    pass_records += 1
                    pairs = flatten_pass_pattern_sentence_pairs(rec, fpath.name, ln)
                    for obj in pairs:
                        append_jsonl_line_eval(fp_pass, obj)
                    pass_lines += len(pairs)
                else:
                    rid = rec.get("id")
                    if isinstance(rid, str):
                        no_pass_ids.add(rid)

                    for r in reasons:
                        global_fail_counter[r] = global_fail_counter.get(r, 0) + 1

                    rec2 = dict(rec)
                    rec2["__eval"] = {"pass": False, "reasons": reasons, "lineno": ln, "source_file": fpath.name}
                    append_jsonl_line_eval(fp_nopass, rec2)

    # no_pass の id を使って origin を回収
    origin_stats = export_no_pass_origin(
        no_pass_ids=no_pass_ids,
        origin_dir=origin_dir,
        origin_glob=origin_glob,
        origin_recursive=origin_recursive,
        out_path=origin_path,
    )

    report = {
        "gen_dir": str(gen_dir),
        "gen_glob": gen_glob,
        "out_dir": str(out_dir),
        "pass_pairs_jsonl": str(pass_path),
        "no_pass_jsonl": str(nopass_path),
        "no_pass_origin_jsonl": str(origin_path),
        "総レコード数": total_records,
        "passレコード数": pass_records,
        "pass（パターン+文）行数": pass_lines,
        "no_passレコード数": total_records - pass_records,
        "no_pass理由集計": global_fail_counter,
        "no_pass_origin回収": origin_stats,
        "no_pass_id数": len(no_pass_ids),
    }
    write_json(report_path, report)

    logger.info(
        "評価完了: 総=%d pass=%d no_pass=%d（id=%d）",
        total_records,
        pass_records,
        total_records - pass_records,
        len(no_pass_ids),
    )

    return {
        "pass_pairs_path": pass_path,
        "no_pass_path": nopass_path,
        "no_pass_origin_path": origin_path,
        "no_pass_ids": no_pass_ids,
        "report": report,
    }


# -------------------------
# ループ本体
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Text→Pattern→Eval→Regen を no_pass=0 まで繰り返す",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ループ制御
    ap.add_argument("--max_iters", type=int, default=10, help="最大反復回数（保険。0なら無制限）")

    # 初回入力（オリジナル）
    ap.add_argument("--origin_dir", type=str, default="../data/no_pass_v3", help="オリジナル ground_truth のディレクトリ")
    ap.add_argument("--origin_glob", type=str, default="no_pass_origin.jsonl", help="オリジナル ground_truth の glob")
    ap.add_argument("--origin_recursive", action="store_true", default=False, help="オリジナル探索を再帰的にする")

    # 生成設定
    ap.add_argument("--schema", type=str, default="../schema/schema.json", help="PatternRepoOutput 用 JSON Schema")
    ap.add_argument("--disable_parallel", action="store_true", default=False, help="並列を無効化（単一Ollama）")
    ap.add_argument("--ollama_base_url", type=str, default="http://ollama:11434", help="単一実行用")
    ap.add_argument(
        "--ollama_urls",
        type=str,
        default="http://ollama0:11434,http://ollama1:11434,http://ollama2:11434,http://ollama3:11434",
        help="並列実行用（カンマ区切り）",
    )
    ap.add_argument("--model", type=str, default="gpt-oss:20b")
    ap.add_argument("--endpoint", type=str, default="chat", choices=["chat", "generate"])
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--think", type=str, default="high")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=1)
    ap.add_argument("--progress_chunk", type=int, default=200)
    ap.add_argument("--keep_tmp", action="store_true", default=False, help="一時ファイルを残す（デバッグ用）")

    # 評価設定
    ap.add_argument("--gen_glob", type=str, default="t2p_*.jsonl", help="生成結果の glob（評価対象）")

    # 出力ルート
    ap.add_argument("--run_dir", type=str, default="../data/", help="実行出力のルートディレクトリ（空なら timestamp）")

    # ログ
    ap.add_argument("--log_level", type=str, default="INFO")

    args = ap.parse_args()
    setup_logger(args.log_level)

    origin_dir = Path(args.origin_dir)
    schema_path = Path(args.schema)

    if not origin_dir.exists():
        raise FileNotFoundError(f"origin_dir が見つかりません: {origin_dir}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema が見つかりません: {schema_path}")

    # run_dir
    if args.run_dir.strip():
        run_root = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = Path("t2p_regen_runs") / ts
    ensure_dir(run_root)

    # 初回入力（複数ファイル）
    origin_files = list_inputs(origin_dir, args.origin_glob, args.origin_recursive)
    if not origin_files:
        raise FileNotFoundError(f"オリジナルが見つかりません: {origin_dir}/{args.origin_glob}")

    # 文節分割器（評価で1回だけ初期化）
    segmenter = BunsetsuSegmenter()
    cfg = EvalConfig()

    # Ollama URL
    ollama_urls = parse_ollama_urls(args.ollama_urls)

    # 最終的な pass を集約（各反復の pass_pairs を順に連結）
    final_pass_path = run_root / "pass_pairs_all.jsonl"
    final_no_pass_origin_path = run_root / "no_pass_origin_final.jsonl"
    final_summary_path = run_root / "summary.json"

    # 反復状態
    current_inputs: List[Path] = origin_files
    total_iters = 0
    last_no_pass = None

    summary_iters: List[Dict[str, Any]] = []

    with final_pass_path.open("w", encoding="utf-8") as fp_final_pass:
        while True:
            total_iters += 1
            iter_dir = run_root / f"iter_{total_iters:02d}"
            gen_out_dir = iter_dir / "gen"
            eval_out_dir = iter_dir / "eval"
            tmp_root = iter_dir / "_tmp"

            ensure_dir(iter_dir)

            logger.info("========== 反復 %d 開始 ==========", total_iters)

            # 生成
            gen_files = run_generation_for_files(
                in_files=current_inputs,
                out_dir=gen_out_dir,
                schema_path=schema_path,
                disable_parallel=args.disable_parallel,
                ollama_base_url=args.ollama_base_url,
                ollama_urls=ollama_urls,
                model=args.model,
                endpoint=args.endpoint,
                max_tokens=args.max_tokens,
                num_ctx=args.num_ctx,
                think=args.think,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
                tmp_root=tmp_root,
                keep_tmp=args.keep_tmp,
                progress_chunk=args.progress_chunk,
            )

            # 評価
            eval_res = run_evaluation_dir(
                gen_dir=gen_out_dir,
                gen_glob=args.gen_glob,
                out_dir=eval_out_dir,
                origin_dir=origin_dir,
                origin_glob=args.origin_glob,
                origin_recursive=args.origin_recursive,
                segmenter=segmenter,
                cfg=cfg,
            )

            # pass を最終ファイルに追記
            pass_pairs_path: Path = eval_res["pass_pairs_path"]
            for _, obj in iter_jsonl_eval(pass_pairs_path):
                append_jsonl_line_eval(fp_final_pass, obj)

            no_pass_ids: Set[str] = eval_res["no_pass_ids"]
            no_pass_origin_path: Path = eval_res["no_pass_origin_path"]

            summary_iters.append({
                "iter": total_iters,
                "inputs": [p.name for p in current_inputs],
                "gen_outputs": [p.name for p in gen_files],
                "no_pass_id数": len(no_pass_ids),
                "eval_report": str(eval_out_dir / "eval_report.json"),
            })

            # 終了条件
            if len(no_pass_ids) == 0:
                logger.info("no_pass が 0 になりました。反復を終了します。")
                break

            # 進捗が止まった場合（同じ no_pass のまま）に警告して保険停止
            if last_no_pass is not None and len(no_pass_ids) == last_no_pass:
                logger.warning("no_pass 数が減っていません（%d）。同じ失敗を繰り返している可能性があります。", len(no_pass_ids))

            last_no_pass = len(no_pass_ids)

            # 次の入力は no_pass_origin.jsonl（単一ファイル）
            if not no_pass_origin_path.exists():
                raise FileNotFoundError(f"no_pass_origin が作成されませんでした: {no_pass_origin_path}")

            current_inputs = [no_pass_origin_path]

            # 最大反復回数（0なら無制限）
            if args.max_iters != 0 and total_iters >= args.max_iters:
                logger.warning("最大反復回数に到達したため終了します（max_iters=%d）。", args.max_iters)
                break

            logger.info("========== 反復 %d 終了（次は no_pass のみ再生成） ==========", total_iters)

    # 最後に no_pass_origin_final を作る（最後の反復で no_pass が残った場合のみ）
    # 最終反復の no_pass_origin をコピー扱いで保存
    #（最終 iter の eval/no_pass_origin.jsonl を指す）
    if summary_iters:
        last_iter_dir = run_root / f"iter_{total_iters:02d}" / "eval"
        last_origin = last_iter_dir / "no_pass_origin.jsonl"
        if last_origin.exists():
            final_no_pass_origin_path.write_text(last_origin.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            final_no_pass_origin_path.write_text("", encoding="utf-8")

    summary = {
        "run_root": str(run_root),
        "pass_pairs_all": str(final_pass_path),
        "no_pass_origin_final": str(final_no_pass_origin_path),
        "iters": summary_iters,
    }
    write_json(final_summary_path, summary)

    logger.info("完了: %s", str(run_root))
    logger.info("最終 pass: %s", str(final_pass_path))
    logger.info("最終 no_pass_origin: %s", str(final_no_pass_origin_path))


if __name__ == "__main__":
    main()
