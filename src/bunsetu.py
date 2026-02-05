# bunsetsu_segmenter.py
from __future__ import annotations
from typing import List, Dict, Any
import spacy, ginza, yaml, pathlib
import time
import logging
# ----------------- 設定読み込み -----------------
HERE = pathlib.Path(__file__).resolve().parent
DICT_PATH = HERE / "parallel_connectives.yml"
with DICT_PATH.open(encoding="utf-8") as f:
    CONNECTIVES_PARALLEL: set[str] = set(yaml.safe_load(f))

BRACKETS = [("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】")]


# spaCy / GiNZA
# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_spacy_model(name, timeout=60):
    logging.info(f"spaCyモデル「{name}」の読み込みを開始します。")
    start = time.time()
    try:
        nlp = spacy.load(name)
    except (IOError, OSError, TypeError) as e:
        logging.error(f"spaCyモデル「{name}」の読み込みに失敗しました: {e}")
        raise
    end = time.time()
    elapsed = end - start
    logging.info(f"spaCyモデル「{name}」の読み込み完了（{elapsed:.1f}秒）")
    if elapsed > timeout:
        logging.error(f"spaCyモデル「{name}」の読み込みに{timeout}秒以上かかりました（{elapsed:.1f}秒）")
        raise TimeoutError(f"spaCyモデル「{name}」の読み込みがタイムアウトしました。")
    return nlp

try:
    nlp = load_spacy_model("ja_ginza_bert_large", timeout=60)
except Exception as e:
    logging.warning(f"BERT GiNZAの読み込みに失敗したので、通常GiNZAで再試行します。: {e}")
    try:
        nlp = load_spacy_model("ja_ginza", timeout=30)
    except Exception as e2:
        logging.error(f"通常GiNZAの読み込みにも失敗しました: {e2}")
        raise

# 読み込み成功後
logging.info(f"使用中のspaCyモデル: {nlp.meta['name']}")


# ------------- 辞書読み込み -----------------
CHUNK_PATH = HERE / "fixed_chunks.yml"
with open('fixed_chunks.yml', encoding='utf-8') as f:
    POST_PATTERNS = [line.split() for line in yaml.safe_load(f)]

# ------------- 汎用マージ関数 ----------------
def _merge_preceding_chunk(spans, post_patterns):
    """
    pattern 直前の span ＋ pattern 部分を 1 文節に。
    末尾トークンが句読点の場合「句読点だけ別 span」でも認識。
    """
    tokens = [tok for sp in spans for tok in sp]
    tok_texts = [t.text for t in tokens]

    # span 開始インデックスをメモ
    span_starts, n = [], 0
    for sp in spans:
        span_starts.append(n)
        n += len(sp)

    merged_flags = [False]*len(tokens)
    merge_ranges = []

    for pattern in post_patterns:
        pat_len = len(pattern)
        idx = 1
        while idx <= len(tokens) - pat_len:
            window = tok_texts[idx:idx+pat_len]
            # パターン末尾が句読点なら "句読点が別 span" パターンも許容
            if window == pattern or (
                pattern[-1] in {"、", "。"} and
                tok_texts[idx:idx+pat_len-1] == pattern[:-1] and
                tok_texts[idx+pat_len-1] in {"、", "。"}
            ):
                start = idx - 1  # 直前トークンごと
                end   = idx + pat_len
                if window != pattern:
                    end += 1      # 句読点だけ別トークン分
                if not any(merged_flags[start:end]):
                    merge_ranges.append((start, end))
                    for j in range(start, end):
                        merged_flags[j] = True
                idx = end
            else:
                idx += 1

    # --- 後段は従来どおり ---
    doc = spans[0].doc
    i, result = 0, []
    while i < len(tokens):
        for s, e in merge_ranges:
            if i == s:
                result.append(doc[tokens[s].i : tokens[e-1].i+1])
                i = e
                break
        else:
            # span 単位で進む
            span_idx = max(k for k, st in enumerate(span_starts) if st <= i)
            span_end = span_starts[span_idx+1] if span_idx+1 < len(span_starts) else len(tokens)
            result.append(doc[tokens[i].i : tokens[span_end-1].i+1])
            i = span_end
    return result




# ------------ ① 単独文節→前に吸収 --------------
def _is_parallel_connective_span(span) -> bool:
    return (
        all(tok.pos_ == "CCONJ" and tok.tag_ == "接続詞" for tok in span)
        and span[0].text in CONNECTIVES_PARALLEL
    )

def _merge_connectives(spans):
    merged = []
    for sp in spans:
        if _is_parallel_connective_span(sp) and merged:
            merged[-1] = sp.doc[merged[-1].start : sp.end]  # 前 span を拡張
        else:
            merged.append(sp)
    return merged
# -------------------------------------------------
# 名詞(及びサ変名詞) 判定
def _is_nominal(tok) -> bool:
    if tok.pos_ in {"NOUN", "PROPN"}:
        return True
    # サ変可能動詞（GiNZA では VERB）も名詞列とみなす
    return tok.pos_ == "VERB" and "サ変可能" in tok.tag_

# -------------------------------------------------

# ------------ ② 同一文節内で再分割 --------------
# span 内で繰り返し分割するバージョン
def _split_span_by_connectors(span, doc):
    """
    接続語が複数あってもすべてで分割し，順序通りに Spans を返す
    例: [整理・管理&運営] → [整理・] [管理&] [運営]
    """
    boundaries = []           # 接続語の直後インデックスを格納
    for i, tok in enumerate(span):
        if tok.text in CONNECTIVES_PARALLEL and tok.pos_ in {"NOUN", "CCONJ", "SYM"}:
            if 0 < i < len(span)-1 and _is_nominal(span[i-1]) and _is_nominal(span[i+1]):
                boundaries.append(tok.i + 1)

    if not boundaries:          # 分割不要
        return [span]

    # boundaries を使って細切れ Spans を作成
    start = span.start
    pieces = []
    for b in boundaries:
        pieces.append(doc[start:b])  # b は token.i+1 なので直後まで含む
        start = b
    pieces.append(doc[start:span.end])  # 残り
    return pieces

def _merge_brackets_and_particle(spans, bracket_pairs=BRACKETS):
    """
    spans リストを走査し，開き括弧〜閉じ括弧(+直後助詞/読点) が
    複数 span に跨っていても 1 span にまとめる。
    """
    merged = []
    open_set  = {l for l, _ in bracket_pairs}
    close_map = {l: r for l, r in bracket_pairs}

    i = 0
    while i < len(spans):
        sp = spans[i]
        # span 先頭トークンが開き括弧？
        first_tok = sp[0]
        if first_tok.text in open_set:
            l_br = first_tok.text
            r_br = close_map[l_br]
            # 開き括弧 span から順に閉じ括弧 span を探す
            j = i
            found = False
            while j < len(spans):
                if any(tok.text == r_br for tok in spans[j]):
                    found = True
                    break
                j += 1
            if found:
                # i〜j まで + 直後の助詞/読点 span も取り込む
                end = j
                if end + 1 < len(spans) and all(
                        tok.pos_ in {"ADP", "PUNCT"} for tok in spans[end+1]):
                    end += 1
                merged_span = sp.doc[spans[i][0].i : spans[end][-1].i + 1]
                merged.append(merged_span)
                i = end + 1
                continue
        # 括弧開始でない → そのまま
        merged.append(sp)
        i += 1
    return merged

def _merge_trailing_comma(spans):
    """
    先頭トークンが読点「、」の span を必ず直前の span に吸収する。
    先頭が「、」で、かつ結合先が存在する場合のみマージ。
    """
    merged = []
    for sp in spans:
        if sp[0].text == "、" and merged:
            # 直前 span の開始〜現在 span の終端で新 span を作成
            prev = merged.pop()
            merged.append(sp.doc[prev.start : sp.end])
        else:
            merged.append(sp)
    return merged


import re

def _merge_alnum_katakana_spans(spans):
    """
    英数字・カタカナの連続をひとまとまりにマージする
    """
    def is_alnum_or_katakana(text):
        # 英字・数字・カタカナにマッチ
        return bool(re.match(r'^[A-Za-z0-9ァ-ンヴー・\'’]+$', text))
    merged = []
    buf = []
    for sp in spans:
        if all(is_alnum_or_katakana(t.text) for t in sp):
            buf.append(sp)
        else:
            if buf:
                # まとめてひとつのspanに
                start = buf[0].start
                end = buf[-1].end
                merged.append(sp.doc[start:end])
                buf = []
            merged.append(sp)
    if buf:
        start = buf[0].start
        end = buf[-1].end
        merged.append(sp.doc[start:end])
    return merged


ENG_RE = re.compile(r"^[A-Za-z0-9'’-]+$")  # ハイフンやアポストロフィも許容

def _merge_consecutive_english_spans(spans):
    """
    連続する英語（英数字、記号、助詞含む）のspanをまとめて1つのspanにする。
    英語以外（日本語名詞や動詞など）が登場した時点で区切る。
    """
    merged = []
    buffer = []
    buffer_start = None
    buffer_end = None
    state = "idle"  # "idle" or "in_english"
    doc = spans[0].doc if spans else None

    for sp in spans:
        # 英語トークンまたは「英語＋助詞/記号」だけの場合
        # -> ただし、先頭は必ず英語で始まる
        tokens = list(sp)
        if (state == "idle" and ENG_RE.match(tokens[0].text)):
            # 連結開始
            buffer_start = sp.start
            buffer_end = sp.end
            buffer = [sp]
            state = "in_english"
        elif state == "in_english" and all(
                ENG_RE.match(tok.text) or tok.pos_ in {"ADP", "PUNCT", "SYM"} for tok in tokens):
            buffer_end = sp.end
            buffer.append(sp)
        else:
            if buffer:
                merged.append(doc[buffer_start:buffer_end])
                buffer = []
                state = "idle"
            merged.append(sp)
    # 最後にバッファが残っていれば出力
    if buffer:
        merged.append(doc[buffer_start:buffer_end])
    return merged


# ---------------- メインクラス --------------------
class BunsetsuSegmenter:
    @staticmethod
    def segment(sentence: str) -> List[List[Any]]:
        doc = nlp(sentence)

        spans = list(ginza.bunsetu_spans(doc))             # ① GiNZA
        spans = _merge_connectives(spans)                  # ② 接続語吸収
        spans = _merge_brackets_and_particle(spans)        # ③ 括弧マージ
        spans = _merge_preceding_chunk(spans, POST_PATTERNS)  # ④ 固定句マージ
        spans = _merge_consecutive_english_spans(spans)    # ★⑤ 英語連結  ← NEW
        spans = _merge_alnum_katakana_spans(spans)         # ⑥ 英数字+カナ結合
        spans = _merge_trailing_comma(spans)               # ⑦ 読点吸収


        # 6. 文節内の接続語で再分割
        refined = []
        for sp in spans:
            refined.extend(_split_span_by_connectors(sp, doc))

        # 4. 出力整形
        bunsetsu_list = [
            [
                sp.text,
                (sp.start_char, sp.end_char),
                [t.text for t in sp],
                [t.pos_ for t in sp],
                [t.tag_ for t in sp],
                [(t.idx, t.idx + len(t.text)) for t in sp],
            ]
            for sp in refined
        ]
        return bunsetsu_list


    @classmethod
    def segment_sentences(cls, sentences: List[str]) -> Dict[str, List[List[Any]]]:
        """
        複数文をまとめて処理し，{sentence: bunsetsu_list} で返す。

        Parameters
        ----------
        sentences : List[str]

        Returns
        -------
        Dict[str, List[List[Any]]]
        """
        docs = nlp.pipe(sentences)
        return {
            sent: cls.segment(sent)  # type: ignore[arg-type]
            for sent, doc in zip(sentences, docs)
        }


# --- 動作確認 ---
if __name__ == "__main__":
    # test = "各製品ラインに関する組織内のコンプライアンス・プログラムの編成、開発、維持、および調整を主導する"
    # test = "信託商品および信託サービスについて、幅広い管理補助と関連業務の調整を行う"
    # test = "Man on Horsebackはフォルカー・シュレンドルフが監督、脚本を務め、1969年1月1日に公開され、ミヒャエル・コールハースを原作としています。"
    test = "「涼宮ハルヒの消失」は、の制作会社が京都アニメーション、脚本家が志茂文彦、監督が石原立也と武本康弘である作品です"
    seg = BunsetsuSegmenter()
    for b in seg.segment(test):
        print(b)
