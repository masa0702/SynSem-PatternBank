#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""text2pattern_batch.py

指定した jsonl（またはディレクトリ内の *.jsonl）を処理し、各行の
  - id
  - sent_ja
  - final_triples
から PatternRepoOutput（patterns）を生成して jsonl として保存する。

改良点:
- 1ファイルを shard（分割）して、複数GPU（=複数Ollamaエンドポイント）へ並列投げ可能
- 親プロセスで進捗を1本表示（ログが壊れない）
- 入力が複数ファイルでも、各ファイルごとに shard→並列→結合

要件:
- sent_ja が無い/空 -> その行はスキップ（出力に書かない）
- final_triples が無い/空 -> その行はスキップ（出力に書かない）
- 出力は入力ファイルごとに分け、ファイル名の先頭に "t2p_" を付ける
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from pattern_repo_client import (
    load_json_file,
    OllamaConfig,
    OllamaJSONSchemaClient,
    PatternGenerator,
)


# =========================
# 入出力
# =========================

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """JSONL（1行1JSON）を逐次読み込む。"""
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} の {ln} 行目が不正なJSONです: {e}") from e


def count_nonempty_lines(path: Path) -> int:
    """進捗バーの total 用（空行は除外）。"""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def split_jsonl_into_shards(in_path: Path, shard_dir: Path, num_shards: int) -> List[Path]:
    """1つのjsonlを num_shards 個に分割（空行はスキップ）。"""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"{in_path.stem}.shard{si:02d}.jsonl" for si in range(num_shards)]

    writers = [p.open("w", encoding="utf-8") for p in shard_paths]
    try:
        idx = 0
        with in_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                si = idx % num_shards
                writers[si].write(line if line.endswith("\n") else line + "\n")
                idx += 1
    finally:
        for w in writers:
            w.close()

    return shard_paths


# =========================
# triples正規化
# =========================

def normalize_final_triples(final_triples: Any) -> List[Dict[str, str]]:
    """final_triples を [{sub, rel, obj}, ...] に正規化（無理なものは空で返す）。"""
    if not isinstance(final_triples, list) or not final_triples:
        return []

    out: List[Dict[str, str]] = []
    for t in final_triples:
        if not isinstance(t, dict):
            continue
        sub = t.get("sub")
        rel = t.get("rel")
        obj = t.get("obj")
        if not (isinstance(sub, str) and isinstance(rel, str) and isinstance(obj, str)):
            continue
        if not (sub.strip() and rel.strip() and obj.strip()):
            continue
        out.append({"sub": sub, "rel": rel, "obj": obj})

    return out


def index_triples(triples: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Tuple[int, List[str]]]]:
    """出力用の indexed dict と、LLM入力用の (idx,[A,P,B]) を作る。"""
    indexed: List[Dict[str, Any]] = []
    triples_with_index: List[Tuple[int, List[str]]] = []

    for i, t in enumerate(triples):
        indexed.append({"idx": i, "sub": t["sub"], "rel": t["rel"], "obj": t["obj"]})
        triples_with_index.append((i, [t["sub"], t["rel"], t["obj"]]))

    return indexed, triples_with_index


# =========================
# shard処理（worker）
# =========================

def _flush_batch(
    items: List[Dict[str, Any]],
    meta: List[Dict[str, Any]],
    wf,
    gen: PatternGenerator,
    *,
    max_workers: int,
) -> None:
    """items/meta を PatternGenerator に流して、結果を jsonl に書く。"""
    try:
        results = gen.generate_batch(items, max_workers=max_workers)
    except Exception as e:
        for m in meta:
            out = {
                "id": m["id"],
                "error": f"バッチ推論に失敗しました: {type(e).__name__}: {e}",
                "final_triples_indexed": m["final_triples_indexed"],
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
        # wf.flush()
        return

    for m, r in zip(meta, results):
        if isinstance(r, dict) and "patterns" in r:
            out = {
                "id": m["id"],
                "sent_ja": m["sent_ja"],
                "final_triples_indexed": m["final_triples_indexed"],
                "patterns": r.get("patterns"),
            }
        else:
            out = {
                "id": m["id"],
                "error": "予期しない推論結果（辞書 or patterns 不在）",
                "final_triples_indexed": m["final_triples_indexed"],
                "raw": r,
            }
        wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    # wf.flush()


def worker_process_shard(
    shard_path: Path,
    part_out_path: Path,
    *,
    schema_path: Path,
    ollama_base_url: str,
    model: str,
    endpoint: str,
    max_tokens: int,
    num_ctx: Optional[int],
    think: str,
    timeout: int,
    max_retries: int,
    max_workers: int,
    progress_q: "mp.Queue[int]",
    progress_chunk: int,
) -> None:
    """1 shard を1 Ollama エンドポイントで処理して part 出力に書く。"""
    schema = load_json_file(str(schema_path))

    cfg = OllamaConfig(
        base_url=ollama_base_url,
        model=model,
        endpoint=endpoint,
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
        timeout_sec=timeout,
        max_retries=max_retries,
        num_ctx=num_ctx,
        think=think,
    )
    client = OllamaJSONSchemaClient(cfg, schema)
    gen = PatternGenerator(
        client,
        prompt_cfg=None,
        enable_bunsetsu=True,
        bunsetsu_module_path="bunsetu.py",
    )

    part_out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_since = 0
    with part_out_path.open("w", encoding="utf-8") as wf:
        batch_items: List[Dict[str, Any]] = []
        batch_meta: List[Dict[str, Any]] = []

        for rec in iter_jsonl(shard_path):
            # 進捗は「読んだ行数（=JSONレコード）」でカウント
            processed_since += 1
            if processed_since >= progress_chunk:
                progress_q.put(processed_since)
                processed_since = 0

            rid = rec.get("id")
            sent_ja = rec.get("sent_ja")
            final_triples = rec.get("final_triples")

            if not (isinstance(rid, str) and rid.strip()):
                continue
            if not (isinstance(sent_ja, str) and sent_ja.strip()):
                continue

            triples_norm = normalize_final_triples(final_triples)
            if not triples_norm:
                continue

            indexed, triples_with_index = index_triples(triples_norm)

            batch_items.append({"id": rid, "sentence": sent_ja, "triples": triples_with_index})
            batch_meta.append({"id": rid, "sent_ja": sent_ja, "final_triples_indexed": indexed})

            # ここは安定優先で「1件=1投げ」
            if len(batch_items) >= 1:
                _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)
                batch_items.clear()
                batch_meta.clear()

        if batch_items:
            _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)

    # 残りの進捗を送る
    if processed_since:
        progress_q.put(processed_since)


# =========================
# 親プロセス：並列実行→結合
# =========================

def concat_jsonl(parts: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for p in parts:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as rf:
                for line in rf:
                    wf.write(line)
    # wf.flush()


def process_one_file_parallel(
    in_path: Path,
    out_path: Path,
    *,
    schema_path: Path,
    ollama_urls: List[str],
    model: str,
    endpoint: str,
    max_tokens: int,
    num_ctx: Optional[int],
    think: str,
    timeout: int,
    max_retries: int,
    max_workers: int,
    tmp_root: Path,
    keep_tmp: bool,
    progress_chunk: int,
) -> None:
    num_shards = len(ollama_urls)
    tmp_dir = tmp_root / f"{in_path.stem}_shards"
    part_dir = tmp_root / f"{in_path.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    # 1) shard分割
    shard_paths = split_jsonl_into_shards(in_path, tmp_dir, num_shards)
    shard_totals = [count_nonempty_lines(p) for p in shard_paths]
    total = sum(shard_totals)

    # 2) worker起動
    ctx = mp.get_context("spawn")
    progress_q: "mp.Queue[int]" = ctx.Queue()

    procs: List[mp.Process] = []
    part_paths: List[Path] = []

    for si, (shard_path, url) in enumerate(zip(shard_paths, ollama_urls)):
        part_out = part_dir / f"{out_path.stem}.part{si:02d}.jsonl"
        part_paths.append(part_out)

        p = ctx.Process(
            target=worker_process_shard,
            kwargs=dict(
                shard_path=shard_path,
                part_out_path=part_out,
                schema_path=schema_path,
                ollama_base_url=url,
                model=model,
                endpoint=endpoint,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
                think=think,
                timeout=timeout,
                max_retries=max_retries,
                max_workers=max_workers,
                progress_q=progress_q,
                progress_chunk=progress_chunk,
            ),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    # 3) 進捗バー（親）
    pbar = tqdm(
        total=total,
        desc=f"{in_path.name} (shards={num_shards})",
        unit="line",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
        mininterval=0.5,
    )

    alive = True
    done_count = 0
    while alive:
        # 進捗を吸い上げ
        try:
            inc = progress_q.get(timeout=0.2)
            pbar.update(int(inc))
        except Exception:
            pass

        done_count = sum(1 for p in procs if not p.is_alive())
        if done_count == len(procs):
            alive = False

    # 念のため join
    for p in procs:
        p.join()

    # 4) 結合
    pbar.close()
    concat_jsonl(part_paths, out_path)

    # 5) tmp掃除
    if not keep_tmp:
        for p in part_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in shard_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # ディレクトリも消せるなら消す
        try:
            part_dir.rmdir()
        except Exception:
            pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


def process_one_file_single(
    in_path: Path,
    out_path: Path,
    *,
    schema_path: Path,
    ollama_base_url: str,
    model: str,
    endpoint: str,
    max_tokens: int,
    num_ctx: Optional[int],
    think: str,
    timeout: int,
    max_retries: int,
    max_workers: int,
) -> None:
    """従来の単一Ollamaでの処理（保険）。"""
    schema = load_json_file(str(schema_path))

    cfg = OllamaConfig(
        base_url=ollama_base_url,
        model=model,
        endpoint=endpoint,
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
        timeout_sec=timeout,
        max_retries=max_retries,
        num_ctx=num_ctx,
        think=think,
    )
    client = OllamaJSONSchemaClient(cfg, schema)
    gen = PatternGenerator(
        client,
        prompt_cfg=None,
        enable_bunsetsu=True,
        bunsetsu_module_path="bunsetu.py",
    )

    total = count_nonempty_lines(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        total=total,
        desc=f"{in_path.name}",
        unit="line",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
        mininterval=0.5,
    )

    with out_path.open("w", encoding="utf-8") as wf:
        batch_items: List[Dict[str, Any]] = []
        batch_meta: List[Dict[str, Any]] = []

        for rec in iter_jsonl(in_path):
            pbar.update(1)

            rid = rec.get("id")
            sent_ja = rec.get("sent_ja")
            final_triples = rec.get("final_triples")

            if not (isinstance(rid, str) and rid.strip()):
                continue
            if not (isinstance(sent_ja, str) and sent_ja.strip()):
                continue

            triples_norm = normalize_final_triples(final_triples)
            if not triples_norm:
                continue

            indexed, triples_with_index = index_triples(triples_norm)

            batch_items.append({"id": rid, "sentence": sent_ja, "triples": triples_with_index})
            batch_meta.append({"id": rid, "sent_ja": sent_ja, "final_triples_indexed": indexed})

            if len(batch_items) >= 1:
                _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)
                batch_items.clear()
                batch_meta.clear()

        if batch_items:
            _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)

    pbar.close()


# =========================
# main
# =========================

def parse_ollama_urls(s: str) -> List[str]:
    urls = [x.strip() for x in s.split(",") if x.strip()]
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()

    # 入力（どちらか）
    ap.add_argument("--in_file", type=str, default=None, help="入力 jsonl ファイル（優先）")
    ap.add_argument("--in_dir", type=str, default="../data/no_pass_v2/", help="入力 jsonl ディレクトリ（*.jsonl を処理）")

    ap.add_argument("--out_dir", type=str, default="../output/ver_3/", help="出力 jsonl ディレクトリ")
    ap.add_argument("--schema", type=str, default="../schema/schema.json", help="PatternRepoOutput 用 JSON Schema")

    # 並列/Ollama
    ap.add_argument("--ollama_base_url", type=str, default="http://ollama:11434", help="単一実行用")
    ap.add_argument(
        "--ollama_urls",
        type=str,
        default="http://ollama0:11434,http://ollama1:11434,http://ollama2:11434,http://ollama3:11434",
        help="並列実行用。カンマ区切りで4つ推奨",
    )
    ap.add_argument("--disable_parallel", action="store_true", help="並列を無効化（単一Ollamaで実行）")

    ap.add_argument("--model", type=str, default="gpt-oss:20b")
    ap.add_argument("--endpoint", type=str, default="chat", choices=["chat", "generate"])
    ap.add_argument("--max_tokens", type=int, default=8192, help="options.num_predict")
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--think", type=str, default="high", help="none/low/medium/high (モデルが対応している場合)")

    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=5)

    # ここは「1プロセス内並列」なので基本1のまま（Ollama側が詰まる）
    ap.add_argument("--max_workers", type=int, default=1)

    # tmp
    ap.add_argument("--tmp_dir", type=str, default=None, help="shard/part の一時置き場（未指定なら out_dir/_tmp）")
    ap.add_argument("--keep_tmp", action="store_true", help="tmp を消さない（デバッグ用）")

    # 進捗キューの頻度（大きいほどオーバーヘッドが減る）
    ap.add_argument("--progress_chunk", type=int, default=200, help="workerが進捗を送る間隔（行数）")

    args = ap.parse_args()

    schema_path = Path(args.schema)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tmp_dir is None:
        tmp_root = out_dir / "_tmp"
    else:
        tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 入力ファイル群を決める
    in_files: List[Path] = []
    if args.in_file is not None:
        in_path = Path(args.in_file)
        if not in_path.exists():
            print(f"入力ファイルが見つかりません: {in_path}", file=sys.stderr, flush=True)
            return
        in_files = [in_path]
    else:
        if args.in_dir is None:
            print("--in_file または --in_dir のどちらかを指定してください。", file=sys.stderr, flush=True)
            return
        in_dir = Path(args.in_dir)
        in_files = sorted(in_dir.glob("*.jsonl"))
        if not in_files:
            print(f"入力 jsonl が見つかりません: {in_dir}", file=sys.stderr, flush=True)
            return

    # 並列設定
    ollama_urls = parse_ollama_urls(args.ollama_urls)
    if len(ollama_urls) < 1:
        print("--ollama_urls が空です。", file=sys.stderr, flush=True)
        return

    for path in in_files:
        out_path = out_dir / f"t2p_{path.name}"
        print(f"\n[処理開始] {path.name} -> {out_path.name}", flush=True)

        if args.disable_parallel or len(ollama_urls) == 1:
            process_one_file_single(
                path,
                out_path,
                schema_path=schema_path,
                ollama_base_url=args.ollama_base_url,
                model=args.model,
                endpoint=args.endpoint,
                max_tokens=args.max_tokens,
                num_ctx=args.num_ctx,
                think=args.think,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
            )
        else:
            process_one_file_parallel(
                path,
                out_path,
                schema_path=schema_path,
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

        print(f"[処理完了] {path.name}", flush=True)


if __name__ == "__main__":
    main()
