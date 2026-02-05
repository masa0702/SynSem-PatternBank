import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from jsonschema import Draft202012Validator

from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path

from pattern_parse_validator import PatternParseValidator


# =========================
# Config
# =========================

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = "gpt-oss:20b"
    endpoint: str = "chat"  # "chat" recommended
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    max_retries: int = 3
    backoff_sec: float = 2.0
    think: str = "medium"  # "none", "low", "medium", "high"

    # generation options
    temperature: float = 0.0
    seed: Optional[int] = 42
    max_tokens: int = 128  # Ollama options.num_predict
    num_ctx: Optional[int] = None  # 使うなら指定


# =========================
# Utilities
# =========================

def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_content_from_ollama_response(data: Dict[str, Any]) -> Any:
    # /api/chat: {"message": {"content": "..."}}
    if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
        return data["message"]["content"]
    # /api/generate: {"response": "..."}
    if "response" in data:
        return data["response"]
    return None


def _extract_thinking_from_ollama_response(data: Dict[str, Any]) -> Optional[str]:
    """/api/chat の thinking フィールド（あれば）を取り出す。"""
    msg = data.get("message")
    if isinstance(msg, dict):
        th = msg.get("thinking")
        if isinstance(th, str):
            return th
    return None


def _coerce_to_json_obj(maybe_obj: Any) -> Tuple[bool, Any]:
    if isinstance(maybe_obj, (dict, list)):
        return True, maybe_obj
    return False, None


def _extract_json_from_text(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"Response content is not a string: {type(text)}")

    s = text.strip()
    if not s:
        raise ValueError("Empty response text (no JSON).")

    # 最速：全体をそのままパース
    if s[0] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    # 混入対策：最初の {/[ と最後の }/] を拾う
    start_candidates = [s.find("{"), s.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        raise ValueError("No JSON object/array start found in response text.")
    start = min(start_candidates)

    end_obj = s.rfind("}")
    end_arr = s.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1 or end < start:
        raise ValueError("No valid JSON end found in response text.")

    snippet = s[start : end + 1]
    return json.loads(snippet)


# =========================
# Ollama JSON Schema Client
# =========================

class OllamaJSONSchemaClient:
    """
    - requests.Session を保持してHTTPコネクションを再利用
    - JSON Schema validator と system prompt を事前生成して使い回す
    """

    def __init__(self, config: OllamaConfig, schema: Dict[str, Any]):
        self.config = config
        self.schema = schema

        self._validator = Draft202012Validator(schema)
        self._system_prompt = self._build_system_prompt(schema)

        self._session = requests.Session()
        self._base_url = config.base_url.rstrip("/")

        endpoint = (config.endpoint or "chat").lower()
        if endpoint not in ("chat", "generate"):
            raise ValueError("endpoint must be 'chat' or 'generate'")
        self._endpoint = endpoint

    @staticmethod
    def _build_system_prompt(schema: Dict[str, Any]) -> str:
        # JSON mode のための最小system（固定スキーマ準拠だけ強制）
        schema_compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        return (
            "あなたは厳密なJSON生成器です。\n"
            "次のJSON Schemaに完全準拠したJSONのみを出力してください。\n"
            "余計な説明文、前置き、コードブロック、Markdown、コメントは一切禁止です。\n"
            "必ず1つのJSONだけを返してください。\n"
            "スキーマ:\n"
            "思考（thinking）や推論過程は出さず、ただちに JSON を返す\n"
            f"{schema_compact}\n"
        )

    def _validate(self, obj: Any) -> None:
        errors = sorted(self._validator.iter_errors(obj), key=lambda e: e.path)
        if errors:
            msgs = []
            for e in errors[:10]:
                loc = "$"
                if e.path:
                    loc += "." + ".".join(map(str, e.path))
                msgs.append(f"{loc}: {e.message}")
            raise ValueError("JSON schema validation failed: " + " | ".join(msgs))

    def _build_payload(
        self,
        user_prompt: str,
        retry_note: Optional[str] = None,
        *,
        think_override: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        opt: Dict[str, Any] = {
            "temperature": float(self.config.temperature),
            "num_predict": int(self.config.max_tokens),
        }
        if self.config.seed is not None:
            opt["seed"] = int(self.config.seed)
        if self.config.num_ctx is not None:
            opt["num_ctx"] = int(self.config.num_ctx)

        # 重要: ユーザープロンプト本文は固定にしたいので、リトライ時の注意書きは system 側に寄せる
        base_system_prompt = self._system_prompt
        retry_note_clean = (retry_note.strip() if retry_note else "")

        if self._endpoint == "chat":
            url = f"{self._base_url}/api/chat"
            messages = [{"role": "system", "content": base_system_prompt}]
            if retry_note_clean:
                messages.append({"role": "system", "content": retry_note_clean})
            messages.append({"role": "user", "content": user_prompt})

            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "format": self.schema,
                "options": opt,
            }
            # thinking 付きモデルはここで明示する（デフォルト挙動がモデル依存で事故る）
            payload["think"] = think_override if think_override is not None else self.config.think
        else:
            url = f"{self._base_url}/api/generate"
            payload = {
                "model": self.config.model,
                "system": base_system_prompt + ("\n" + retry_note_clean if retry_note_clean else ""),
                "prompt": user_prompt,
                "stream": False,
                "format": self.schema,
                "options": opt,
            }
            payload["think"] = think_override if think_override is not None else self.config.think

        return url, payload

    def generate(
        self,
        user_prompt: str,
        debug_dump_path: Optional[str] = None,
        *,
        retry_note: Optional[str] = None,
    ) -> Any:
        """JSON Schemaに準拠したJSONを返す。失敗時はリトライ。"""

        last_err: Optional[Exception] = None
        last_preview = ""

        # thinking モードで content が空のまま終わる事故があるため、
        # その場合は think を弱めて再試行する（プロンプト本文は固定のまま）。
        think_override: Optional[str] = None

        def _next_think_level(cur: str) -> str:
            cur = (cur or "").strip().lower()
            order = ["high", "medium", "low", "none"]
            if cur not in order:
                # 未知は安全側へ
                return "none"
            i = order.index(cur)
            return order[min(i + 1, len(order) - 1)]

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # attempt ごとに payload を組み直す（think のフォールバックを反映する）
                url, payload = self._build_payload(
                    user_prompt,
                    retry_note=retry_note,
                    think_override=think_override,
                )

                resp = self._session.post(url, json=payload, timeout=self.config.timeout_sec)
                resp.raise_for_status()
                data = resp.json()

                if debug_dump_path:
                    # 速度優先でindent無し
                    with open(debug_dump_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps({"request": payload, "response": data}, ensure_ascii=False))

                content = _extract_content_from_ollama_response(data)
                if content is None:
                    raise ValueError(f"Unexpected response keys: {list(data.keys())}")

                # thinking が返ってきているのに content が空 (= JSON が1文字も無い) ケース
                if isinstance(content, str) and not content.strip():
                    thinking = _extract_thinking_from_ollama_response(data)
                    done_reason = data.get("done_reason")
                    raise ValueError(
                        "Empty response content (no JSON). "
                        f"done_reason={done_reason} thinking_present={bool(thinking)}"
                    )

                ok, obj = _coerce_to_json_obj(content)
                if not ok:
                    obj = _extract_json_from_text(str(content))

                self._validate(obj)
                return obj

            except Exception as e:
                last_err = e
                try:
                    last_preview = (resp.text[:300] if "resp" in locals() else "")  # type: ignore
                    last_preview = last_preview.replace("\n", "\\n")
                except Exception:
                    pass

                # content が空で落ちた場合は、think を弱めて次を試す
                msg = str(e)
                if "Empty response content" in msg or "Empty response text" in msg:
                    cur = think_override if think_override is not None else self.config.think
                    think_override = _next_think_level(str(cur))
                    # system 追加1行だけで指示を強める（ユーザープロンプト本文は触らない）
                    if retry_note is None:
                        retry_note = "thinking を短くし、必ず message.content に JSON を出力してください。"
                    else:
                        # 追記は最小
                        if "message.content" not in retry_note:
                            retry_note = retry_note.rstrip() + "\n" + "必ず message.content に JSON を出力してください。"

                if attempt < self.config.max_retries:
                    time.sleep(self.config.backoff_sec * (2 ** (attempt - 1)))
                else:
                    break

        raise RuntimeError(
            f"Failed after {self.config.max_retries} attempts. Last error: {last_err}. "
            f"Response preview: {last_preview}"
        ) from last_err


# =========================
# Prompt builder (Pattern)
# =========================

@dataclass(frozen=True)
class PatternPromptConfig:
    """
    ルール全文を毎回貼らず、短縮ルール＋入力だけにしてトークンを抑える。
    必要なら rule_text を差し替え可能にする。
    """
    rule_text: str = (
        """
        あなたは、日本語文と抽出したいTriple群から「文節ベースのパターン」を生成する。

        【重要: 表層一致】
        - 出力パターンは、入力文の表層順序（機能語の並び）と一致していなければならない。
        - 助詞・助動詞・コピュラ等の機能語は、必ず [] の外に書く。
        - [] の中には X/Y/M/G と修飾メタ記号（*n, #n）および並列グループのみを書く。
        - 不要な空白は禁止（例: [*1X1] のように連結し、[ *1 X1 ] のようにしない）。
        - Tripleを抽出するための最小パターンを作成しなければならない。

        【変数】
        - [Xi] は述語の項（Argument）を表す。subject/object の区別はしない。
        - [Yi] は述語を表す。
        - [Mi] は修飾語（内容語）を表す。
        - 修飾子 *n(連体修飾)/#n(連用修飾) は前置し、右から左へ適用する。
        - i >= 0から始める。

        【ギャップ G（スキップ）】
        - [G{m,n}] は、連続する文節を m〜n個 スキップする（非捕捉。抽出対象には含めない）。
        - 上限 n は必ず指定する（無制限スキップは禁止）。
        - m,n は 0 以上の整数で、m <= n。
        - ギャップ長は 必要最小限にする（過剰に大きくしない）。
        - [G{…}][G{…}] の連続は禁止（1つのギャップに統合する）。
        - パターンの 先頭・末尾にギャップを置かない（アンカー欠落を防ぐ）。
        - ギャップの前後には、少なくとも片側にリテラル（助詞等）または [Yi] などの アンカーを含める。

        【並列 &】
        - & は並列構造の省略表現である。
        - 入力文に並列マーカー（と/や/、/および/または 等）が実在する場合のみ & を使ってよい。
        - 並列グループは必ず ( ... ) で括る。例: ([M1]&[M2])

        【出力（カバー関係）】
        - 出力はパターンのみ。
        - patterns は配列。各要素は以下を含む:
        - covers: このパターンがカバーするTripleのindex配列（0始まり）
        - pattern: パターン文字列

        【最適化】
        - 入力Triple全体を漏れなくカバーせよ（カバー漏れ禁止）。
        - Triple抽出に関係しない文節や文字列はパターンに含めてはならない（過剰禁止）。
        - ただし、Triple要素が文中で分断されている場合、その間の文節はギャップでスキップする。
        - *並列化できる場合のみ*、複数Tripleを1パターンにまとめて pattern 数を小さくする。
        - ただし、表層一致を壊す“無理なまとめ”は禁止。
        - 並列でない場合は、1パターン1Tripleにせよ。
        - 変数要素[]とギャップ要素[G{n,m}]はTriple抽出に関係する文節の数と同一にせよ（過不足禁止）。

        【例1】
        文：Benelli Argo ELは、ベネリ社がデザインしたライフルである。
        文節(Bi):
        B0=Benelli Argo ELは、
        B1=ベネリ社が
        B2=デザインした
        B3=ライフルである。
        Triples（index付き）：
        0: [Benelli Argo EL, デザイン, ベネリ]
        パターン：[X0]は[X1]が[Y0]した

        【例2】
        文：exscientologykids.comは、2008年にJenna Miscavige Hill、Kendra Wisemanが作者として立ち上げられたウェブサイトです。
        文節(Bi):
        B0=exscientologykids.comは、
        B1=2008年に
        B2=Jenna Miscavige Hill、
        B3=Kendra Wisemanが
        B4=作者として
        B5=立ち上げられた
        B6=ウェブサイトです。
        Triples（index付き）：
        0: [exscientologykids.comKendra, 作者, Wiseman]
        1: [exscientologykids.comKendra, 作者, Jenna Miscavige Hill]
        パターン：[X0]は[G{1,2}][X1]&[X2]が[Y0]として

        【入力】

        """
    )

    # 返却を短くしたいので、出力指示も短い固定にする
    output_text: str = "出力はJSONのみ。patternsのみを返す。"


def build_pattern_prompt(
    sentence: str,
    triples_with_index: List[Tuple[int, List[str]]],
    cfg: Optional[PatternPromptConfig] = None,
    bunsetsu_texts: Optional[List[str]] = None,
) -> str:
    """
    triples_with_index: [(0, [A,P,B]), (1, [...]), ...]
    bunsetsu_texts: ["B0の表層", "B1の表層", ...]
    """
    if cfg is None:
        cfg = PatternPromptConfig()

    lines = [cfg.rule_text, "入力:"]
    lines.append(f"文={sentence}")

    # 文節列を追加（トークン増を抑えるため、表層のみ・番号付きで短く）
    if bunsetsu_texts:
        lines.append("文節:")
        for i, bt in enumerate(bunsetsu_texts):
            # 余計な空白を入れない（あなたの規約に合わせる）
            lines.append(f"B{i}={bt}")

    lines.append("Triples:")
    for idx, tri in triples_with_index:
        lines.append(f"{idx}:[{tri[0]},{tri[1]},{tri[2]}]")

    lines.append(cfg.output_text)
    return "\n".join(lines)



# =========================
# Pattern Generator
# =========================
class PatternGenerator:
    """
    - schema は PatternRepoOutput（patterns: [{covers, pattern}]）想定
    - generate_one / generate_batch を提供
    - bunsetu.py による文節化結果を入力に含める
    """
    def __init__(
        self,
        client: OllamaJSONSchemaClient,
        prompt_cfg: Optional[PatternPromptConfig] = None,
        *,
        enable_bunsetsu: bool = True,
        bunsetsu_module_path: Optional[str] = None,
        # 追加：パーサ検証
        enable_parse_check: bool = True,
        parser_path: str = "../pattern_grammar/pattern_parser.py",
        parse_check_max_regen: int = 2,  # パースNG時の再生成回数
    ):
        self.client = client
        self.prompt_cfg = prompt_cfg or PatternPromptConfig()

        self._enable_bunsetsu = bool(enable_bunsetsu)
        self._bunsetu: Optional[BunsetsuProvider] = None
        if self._enable_bunsetsu:
            self._bunsetu = BunsetsuProvider(module_path=bunsetsu_module_path)

        self._parse_validator: Optional[PatternParseValidator] = None
        if enable_parse_check:
            self._parse_validator = PatternParseValidator(parser_path=parser_path)

        self._parse_check_max_regen = int(parse_check_max_regen)

    def generate_one(
        self,
        sentence: str,
        triples_with_index: List[Tuple[int, List[str]]],
        debug_dump_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        bunsetsu_texts: Optional[List[str]] = None
        if self._bunsetu is not None:
            bunsetsu_texts = self._bunsetu.bunsetsu_texts(sentence)

        prompt = build_pattern_prompt(
            sentence,
            triples_with_index,
            self.prompt_cfg,
            bunsetsu_texts=bunsetsu_texts,
        )

        last_errors = ""
        for k in range(self._parse_check_max_regen + 1):
            retry_note = None
            if k > 0:
                # プロンプト本文は変更しない。追加メッセージのみ。
                retry_note = (
                    "前回出力の pattern が PatternParser でパース不能でした。"
                    "必ずパース可能な pattern のみを含むJSONを再出力してください。"
                    f" errors={last_errors}"
                )

            obj = self.client.generate(prompt, debug_dump_path=debug_dump_path, retry_note=retry_note)

            if self._parse_validator is None:
                return obj

            ok, errors = self._parse_validator.validate_output(obj)
            if ok:
                return obj

            # エラーを短く（トークン節約）。先頭2～3件だけ。
            parts = []
            for e in errors[:3]:
                parts.append(f"(i={e.index}) {e.pattern} :: {e.message}")
            last_errors = " | ".join(parts)

        raise RuntimeError(
            "JSON生成は成功しましたが、PatternParserの検証に通りませんでした。"
            f" 最終エラー: {last_errors}"
        )

    def generate_batch(
        self,
        items: Iterable[Dict[str, Any]],
        *,
        max_workers: int = 1,
        debug_dump_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        items の各要素は以下を想定:
        {
            "sentence": str,
            "triples": List[Tuple[int, List[str]]]
        }

        - max_workers=1: 逐次（最も安定）
        - max_workers>1: 並列（Ollama/GPUが許す範囲で）

        debug_dump_dir を指定すると、各入力ごとにデバッグJSONを保存します。
        """
        # items が generator の可能性があるため、必ず list 化してから使う（消費事故を防ぐ）
        items_list = list(items)

        # 逐次は単純に generate_one を回す（ここで parse check / regen が必ず適用される）
        if max_workers <= 1:
            out: List[Dict[str, Any]] = []
            for i, it in enumerate(items_list):
                dump_path = None
                if debug_dump_dir:
                    os.makedirs(debug_dump_dir, exist_ok=True)
                    dump_path = os.path.join(debug_dump_dir, f"debug_{i}.json")

                out.append(self.generate_one(it["sentence"], it["triples"], debug_dump_path=dump_path))
            return out

        # -------------------------
        # 並列実行
        # -------------------------
        # requests.Session はスレッドセーフではないため、各スレッドで client を作る
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cfg = self.client.config
        schema = self.client.schema

        # この PatternGenerator の設定を“そのまま”引き継ぐために、必要な属性を保持しておく
        prompt_cfg = self.prompt_cfg

        enable_bunsetsu = (self._bunsetu is not None)
        bunsetsu_module_path = None
        if enable_bunsetsu and hasattr(self._bunsetu, "module_path"):
            bunsetsu_module_path = getattr(self._bunsetu, "module_path")

        enable_parse_check = (self._parse_validator is not None)
        parser_path = None
        if enable_parse_check and hasattr(self._parse_validator, "_parser"):
            # PatternParseValidator には parser_path を保持していない実装もあり得るため、
            # PatternGenerator 側で保持している前提にするのが確実です。
            # （後述の「小修正」を入れてください）
            parser_path = getattr(self, "_parser_path", None)

        parse_check_max_regen = getattr(self, "_parse_check_max_regen", 0)

        def _worker(i: int, it: Dict[str, Any]) -> Dict[str, Any]:
            local_client = OllamaJSONSchemaClient(cfg, schema)
            local_gen = PatternGenerator(
                local_client,
                prompt_cfg=prompt_cfg,
                enable_bunsetsu=enable_bunsetsu,
                bunsetsu_module_path=bunsetsu_module_path,
                enable_parse_check=enable_parse_check,
                parser_path=parser_path or "../pattern_grammar/pattern_parser.py",
                parse_check_max_regen=parse_check_max_regen,
            )

            dump_path = None
            if debug_dump_dir:
                os.makedirs(debug_dump_dir, exist_ok=True)
                dump_path = os.path.join(debug_dump_dir, f"debug_{i}.json")

            return local_gen.generate_one(it["sentence"], it["triples"], debug_dump_path=dump_path)

        results: List[Optional[Dict[str, Any]]] = [None] * len(items_list)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker, i, it): i for i, it in enumerate(items_list)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

        return [r for r in results if r is not None]

class BunsetsuProvider:
    """
    bunsetu.py の BunsetsuSegmenter を読み込み、文 -> 文節表層列[str] を返す。
    - モデル読み込みは import 時に発生し得るので、プロセス内で1回に抑える
    - 文章の重複がある場合に備え、LRUキャッシュで再計算を回避
    """

    def __init__(self, module_path: Optional[str] = None):
        self._seg = self._load_segmenter(module_path)

    @staticmethod
    def _load_segmenter(module_path: Optional[str]):
        """
        module_path が None の場合:
          - Python import で `bunsetu` を探す（/workspace/src がPYTHONPATHにある想定）
        module_path が指定される場合:
          - そのファイルパスから動的 import
        """
        if module_path:
            p = Path(module_path)
            if not p.exists():
                raise FileNotFoundError(f"bunsetu module not found: {module_path}")
            spec = importlib.util.spec_from_file_location("bunsetu_dynamic", str(p))
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module spec from: {module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        else:
            mod = importlib.import_module("bunsetu")

        if not hasattr(mod, "BunsetsuSegmenter"):
            raise AttributeError("bunsetu.py に BunsetsuSegmenter が見つかりません。")

        Seg = getattr(mod, "BunsetsuSegmenter")
        # segment は @staticmethod なのでインスタンス化不要だが、呼びやすいように保持
        return Seg

    @lru_cache(maxsize=8192)
    def bunsetsu_texts(self, sentence: str) -> List[str]:
        """
        bunsetu.py の segment(sentence) の返り値は
        [[sp.text, (start,end), ...], ...] なので、先頭の sp.text だけ使う。
        """
        bun = self._seg.segment(sentence)
        # 安全策：形式が崩れていても落ちにくくする
        out: List[str] = []
        for row in bun:
            if isinstance(row, (list, tuple)) and len(row) >= 1:
                out.append(str(row[0]))
        return out
