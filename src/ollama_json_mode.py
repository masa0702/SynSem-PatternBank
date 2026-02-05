import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests
from jsonschema import Draft202012Validator


DEFAULT_BASE_URL = "http://ollama:11434"
DEFAULT_TIMEOUT_SEC = 300  # 初回は重いことがあるため長め


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_system_prompt(schema: Dict[str, Any]) -> str:
    schema_compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "あなたは厳密なJSON生成器です。\n"
        "次のJSON Schemaに完全準拠したJSONのみを出力してください。\n"
        "余計な説明文、前置き、コードブロック、Markdown、コメントは一切禁止です。\n"
        "必ず1つのJSONオブジェクト（スキーマが配列ならJSON配列）だけを返してください。\n"
        "スキーマ:\n"
        f"{schema_compact}\n"
    )


def _validate_json(schema: Dict[str, Any], obj: Any) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors[:10]:
            loc = "$"
            if e.path:
                loc += "." + ".".join(map(str, e.path))
            msgs.append(f"{loc}: {e.message}")
        raise ValueError("JSON schema validation failed: " + " | ".join(msgs))


def _coerce_to_json_obj(maybe_obj: Any) -> Tuple[bool, Any]:
    """
    返却が既に dict/list ならそのまま返す。
    """
    if isinstance(maybe_obj, (dict, list)):
        return True, maybe_obj
    return False, None


def _extract_json_from_text(text: str) -> Any:
    """
    テキストからJSON（object/array）を抽出して json.loads する。
    schema強制が効いていれば text 自体が JSON のはずだが、
    念のため前後に余計な文字が混入した場合も拾う。
    """
    if not isinstance(text, str):
        raise ValueError(f"Response content is not a string: {type(text)}")

    s = text.strip()
    if not s:
        raise ValueError("Empty response text (no JSON).")

    # まず全体をそのままJSONとして試す（最速）
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


def _extract_content_from_ollama_response(data: Dict[str, Any]) -> Any:
    """
    /api/chat と /api/generate の両方に対応して「中身」を取り出す。
    - chat: data["message"]["content"]
    - generate: data["response"]
    """
    if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
        return data["message"]["content"]
    if "response" in data:
        return data["response"]
    return None


def _dump_debug(debug_dump: Optional[str], payload: Dict[str, Any], data: Any) -> None:
    if not debug_dump:
        return
    try:
        with open(debug_dump, "w", encoding="utf-8") as f:
            f.write("### REQUEST PAYLOAD\n")
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            f.write("\n\n### RESPONSE JSON\n")
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.write("\n")
    except Exception:
        # デバッグ保存失敗は本処理を止めない
        pass


def ollama_generate_json(
    base_url: str,
    model: str,
    prompt: str,
    schema: Dict[str, Any],
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = 3,
    backoff_sec: float = 2.0,
    endpoint: str = "chat",
    debug_dump: Optional[str] = None,
) -> Any:
    """
    Ollamaに固定スキーマJSONを返させる。
    デフォルトは /api/chat（structured output が安定しやすい）。
    """
    system = _build_system_prompt(schema)
    session = requests.Session()

    endpoint = endpoint.lower()
    if endpoint not in ("chat", "generate"):
        raise ValueError("--endpoint must be 'chat' or 'generate'")

    if endpoint == "chat":
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
            },
        }
    else:
        url = base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
            },
        }

    if seed is not None:
        payload["options"]["seed"] = seed

    last_err: Optional[Exception] = None
    last_text_preview: str = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()

            _dump_debug(debug_dump, payload, data)

            content = _extract_content_from_ollama_response(data)
            if content is None:
                raise ValueError(f"Unexpected response keys: {list(data.keys())}")

            # 既に dict/list で返る場合はそのまま
            ok, obj = _coerce_to_json_obj(content)
            if not ok:
                # 文字列ならJSON抽出してパース
                obj = _extract_json_from_text(str(content))

            _validate_json(schema, obj)
            return obj

        except Exception as e:
            last_err = e
            try:
                # 返却本文の先頭だけでも残す（原因特定用）
                if "resp" in locals():
                    last_text_preview = resp.text[:300].replace("\n", "\\n")
            except Exception:
                pass

            if attempt < max_retries:
                time.sleep(backoff_sec * (2 ** (attempt - 1)))
            else:
                break

    raise RuntimeError(
        f"Failed after {max_retries} attempts. Last error: {last_err}. "
        f"Response preview: {last_text_preview}"
    ) from last_err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama (gpt-oss) strict JSON Schema client (chat/generate)."
    )
    parser.add_argument("--model", required=False, default="gpt-oss:20b", help="Ollama model name (e.g., gpt-oss:20b)")
    parser.add_argument("--schema", required=False, default="../schema/schema.json", help="Path to JSON Schema file (fixed schema)")
    parser.add_argument("--prompt", default=None, help="Prompt text (if omitted, read from --prompt_file or stdin)")
    parser.add_argument("--prompt_file", default=None, help="Read prompt from file")
    parser.add_argument("--base_url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL), help="Ollama base URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="HTTP timeout (sec)")
    parser.add_argument("--retries", type=int, default=3, help="Max retries on failure")
    parser.add_argument("--endpoint", default="chat", help="chat (recommended) or generate")
    parser.add_argument("--debug_dump", default=None, help="Write request/response to file for debugging")
    parser.add_argument("--out", default="../output/output.json", help="Output path for JSON (optional)")

    args = parser.parse_args()

    schema = _load_json(args.schema)

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = sys.stdin.read()

    result = ollama_generate_json(
        base_url=args.base_url,
        model=args.model,
        prompt=prompt,
        schema=schema,
        temperature=args.temperature,
        seed=args.seed,
        timeout_sec=args.timeout,
        max_retries=args.retries,
        endpoint=args.endpoint,
        debug_dump=args.debug_dump,
    )

    out_text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
