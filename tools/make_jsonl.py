import argparse
import json
from typing import Iterable


def iter_rows_tsv(fp: Iterable[str]):
    header = None
    for line_no, raw in enumerate(fp, start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        cols = line.split("\t")
        if header is None:
            header = cols
            # 最低限、id と pattern があるか確認
            if "id" not in header or "pattern" not in header:
                raise ValueError(f"ヘッダに 'id' と 'pattern' が必要です: {header}")
            id_idx = header.index("id")
            pat_idx = header.index("pattern")
            continue

        # 列が足りない行はスキップ or エラーにしたいなら raise に変更
        if len(cols) <= max(id_idx, pat_idx):
            raise ValueError(f"{line_no}行目: 列数が不足しています: {cols}")

        rid = cols[id_idx].strip()
        pat = cols[pat_idx].strip()

        if not rid or not pat:
            # 空は無視（必要ならエラーに）
            continue

        yield {"id": rid, "pattern": pat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", default="patterns.tsv", help="入力TSV（タブ区切り）")
    ap.add_argument("--out_jsonl", default="patterns_from_manually.jsonl", help="出力JSONL")
    ap.add_argument("--ensure_ascii", action="store_true", help="ASCIIエスケープを有効化（通常は不要）")
    args = ap.parse_args()

    with open(args.in_tsv, "r", encoding="utf-8") as rf, open(args.out_jsonl, "w", encoding="utf-8") as wf:
        for obj in iter_rows_tsv(rf):
            wf.write(json.dumps(obj, ensure_ascii=args.ensure_ascii) + "\n")


if __name__ == "__main__":
    main()
