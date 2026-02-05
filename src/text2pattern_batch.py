#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""text2pattern_batch.py

指定ディレクトリ内の *.jsonl を全て処理し、各行の
  - id
  - sent_ja
  - final_triples
から PatternRepoOutput（patterns）を生成して、別の jsonl として保存する。

要件:
- sent_ja が無い/空 -> その行はスキップ（出力に書かない）
- final_triples が無い/空 -> その行はスキップ（出力に書かない）
- 出力は入力ファイルごとに分け、ファイル名の先頭に "t2p_" を付ける
- 処理中のファイル名と進捗が分かるように tqdm で表示する

注意:
- Docker logs 等の非TTY環境だと tqdm のバーが「動かない」ように見えることがある。
  その場合は `python -u text2pattern_batch.py ...` で実行し、標準出力のバッファリングを切る。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from pattern_repo_client import (
    load_json_file,
    OllamaConfig,
    OllamaJSONSchemaClient,
    PatternGenerator,
    BunsetsuProvider,
)


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


def count_lines(path: Path) -> int:
    """進捗バーの total 用（ざっくりでOK）。"""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


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
    for i, t in enumerate(triples):
        indexed.append({"idx": i, "sub": t["sub"], "rel": t["rel"], "obj": t["obj"]})

    triples_with_index: List[Tuple[int, List[str]]] = []
    for i, t in enumerate(triples):
        triples_with_index.append((i, [t["sub"], t["rel"], t["obj"]]))

    return indexed, triples_with_index


def process_one_file(
    in_path: Path,
    out_path: Path,
    *,
    gen: PatternGenerator,
    max_workers: int,
) -> None:
    """1つの入力 jsonl を処理して、出力 jsonl を生成する。"""

    total = count_lines(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 非TTYでも tqdm を出したいので disable=False
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

        # まず入力を読みながらバッチを作る
        for rec in iter_jsonl(in_path):
            pbar.update(1)

            rid = rec.get("id")
            sent_ja = rec.get("sent_ja")
            final_triples = rec.get("final_triples")

            # 要件: sent_ja / final_triples のどちらかが無い or 空ならスキップ
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

            # max_workers>1 の並列は、Ollama 側が OLLAMA_NUM_PARALLEL=1 なら効かない。
            # ここでは 1 件ずつ流す（安定優先）。
            if len(batch_items) >= 1:
                _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)
                batch_items.clear()
                batch_meta.clear()

        # 残り
        if batch_items:
            _flush_batch(batch_items, batch_meta, wf, gen, max_workers=max_workers)

    pbar.close()


def _flush_batch(
    items: List[Dict[str, Any]],
    meta: List[Dict[str, Any]],
    wf,
    gen: PatternGenerator,
    *,
    max_workers: int,
) -> None:
    """items/meta を PatternGenerator に流して、結果を jsonl に書く。"""

    # generate_batch は max_workers で並列できるが、まずは安定重視。
    try:
        results = gen.generate_batch(items, max_workers=max_workers)
    except Exception as e:
        # バッチ丸ごと落ちた場合でも、行単位で原因追跡できるように残す
        for m in meta:
            out = {
                "id": m["id"],
                "error": f"バッチ推論に失敗しました: {type(e).__name__}: {e}",
                "final_triples_indexed": m["final_triples_indexed"],
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
        wf.flush()
        return

    # items と results は同順
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

    wf.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="../data/no_pass_v1/", help="入力 jsonl ディレクトリ")
    ap.add_argument("--out_dir", type=str, default="../output/ver_2/", help="出力 jsonl ディレクトリ")
    ap.add_argument("--schema", type=str, default="../schema/schema.json", help="PatternRepoOutput 用 JSON Schema")

    ap.add_argument("--ollama_base_url", type=str, default="http://ollama:11434")
    ap.add_argument("--model", type=str, default="gpt-oss:20b")
    ap.add_argument("--endpoint", type=str, default="chat", choices=["chat", "generate"])

    ap.add_argument("--max_tokens", type=int, default=8192, help="options.num_predict")
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--think", type=str, default="high", help='Ollama API think: none/low/medium/high (モデルが対応している場合)')

    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=1)

    ap.add_argument("--bunsetu_py", type=str, default="bunsetu.py", help="文節化スクリプト")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    schema_path = Path(args.schema)

    schema = load_json_file(str(schema_path))

    cfg = OllamaConfig(
        base_url=args.ollama_base_url,
        model=args.model,
        endpoint=args.endpoint,
        temperature=0.0,
        seed=42,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout,
        max_retries=args.max_retries,
        num_ctx=args.num_ctx,
        think=args.think,
    )

    client = OllamaJSONSchemaClient(cfg, schema)

    bunsetsu_provider = BunsetsuProvider(module_path=str(Path(args.bunsetu_py)))
    gen = PatternGenerator(
        client,
        prompt_cfg=None,
        enable_bunsetsu=True,
        bunsetsu_module_path=args.bunsetu_py
    )


    in_files = sorted(in_dir.glob("*.jsonl"))
    if not in_files:
        print(f"入力 jsonl が見つかりません: {in_dir}", file=sys.stderr, flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for path in in_files:
        out_path = out_dir / f"t2p_{path.name}"
        print(f"\n[処理開始] {path.name} -> {out_path.name}", flush=True)
        process_one_file(path, out_path, gen=gen, max_workers=args.max_workers)
        print(f"[処理完了] {path.name}", flush=True)


if __name__ == "__main__":
    main()
