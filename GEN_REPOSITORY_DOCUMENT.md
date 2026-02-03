# build_repo_from_patterns_from_text.py ドキュメント

## 1. 目的と位置づけ
`tmp/src/build_repo_from_patterns_from_text.py` は、`patterns_from_text.jsonl` を入力として「パターンリポジトリ」を生成するバッチです。具体的には、以下を作成・更新します。

- `patterns/JA_T2KGB/*.yaml`（個別パターンのYAML）
- `index/patterns.jsonl`（検索向けのJSONL）
- `index/patterns.index.json`（軽量索引のJSON）
- `index/MANIFEST.json`（入出力・統計のメタ情報）

処理は **一旦 `tmp/` に中間生成物を作成** し、最後に `patterns/` と `index/` を置換更新します。

## 2. 入出力ファイル・ディレクトリ
### 入力
- `../patterns_from_text.jsonl`（既定）
  - `--input` で変更可能。
  - 各行はJSON。キーは `pattern` が標準ですが、`pattern_str / pattern_text / pattern_raw` も許容。

### 依存ディレクトリ（必須）
- `../pattern_grammar`（既定）
  - `pattern_parser.py` が必要。
  - `grammar.lark` が存在する場合は `MANIFEST.json` にハッシュを記録。

### 出力
- `patterns/JA_T2KGB/*.yaml`
- `index/patterns.jsonl`
- `index/patterns.index.json`
- `index/MANIFEST.json`

### 中間・レポート
- `tmp/_stage_unique.jsonl`
- `tmp/bad_records.jsonl`
- `tmp/build_report.json`
- `tmp/_patterns.jsonl`
- `tmp/_patterns.index.json`
- `tmp/_MANIFEST.json`

## 3. 起動方法と引数
```
python tmp/src/build_repo_from_patterns_from_text.py [options]
```

主な引数：
- `--input`：入力JSONL（既定 `../patterns_from_text.jsonl`）
- `--tmp-dir`：中間ディレクトリ（既定 `tmp`）
- `--patterns-dir`：YAML出力先（既定 `patterns/JA_T2KGB`）
- `--index-dir`：index出力先（既定 `index`）
- `--pattern-key`：入力JSONのパターン文字列キーを明示指定
- `--status`：`draft | verified | deprecated`（既定 `draft`）
- `--id-prefix`：`pattern_id` の接頭辞（既定 `JA_T2KGB_`）
- `--id-hash-len`：ハッシュの使用長（既定 `12`）
- `--keep-unparsable`：パース失敗でも採用する
- `--parser-dir`：`pattern_parser.py` のあるディレクトリ（既定 `../pattern_grammar`）

## 4. 処理フロー（詳細）
### 4.1 1st pass: streamingで重複除去
入力JSONLを読みながら、以下の条件で重複・不正を排除します。

1. **JSONが壊れている**
   - `bad_records.jsonl` に記録し除外。
2. **pattern文字列が欠落**
   - `bad_records.jsonl` に記録し除外。
3. **リテラルが含まれない**
   - ダブルクオートで囲まれたリテラル（`"..."`）が無い場合は除外。
   - `bad_records.jsonl` に `reason=no_literal` で記録。
4. **X/Y要素が横並び（無連結）**
   - `[X*][X*]` / `[Y*][Y*]` / `[X*][Y*]` / `[Y*][X*]` のように、
     `&`・リテラル・`[G]` なしで直接連結している場合は除外。
   - `bad_records.jsonl` に `reason=adjacent_xy` で記録。
5. **文字列完全一致の重複**
   - 既出と一致したら除外。
6. **AST構造が同一の重複**
   - `pattern_parser` でパースし、ASTを正規化してハッシュ化。
   - 既出ASTハッシュと一致したら除外。
   - Xi/Mi/Yi は **別物として扱う**（ASTの差分があれば別扱い）。
7. **パース失敗**
   - `--keep-unparsable` が **未指定** なら除外して `bad_records.jsonl` に記録。
   - 指定時は採用し、`ast_sig` は `null` 扱い。

採用したレコードは `tmp/_stage_unique.jsonl` に保存されます。

### 4.2 AST正規化と署名
ASTは以下の方針で **構造を揃えた上でSHA256** を計算します。
- `dataclass` → `asdict()`
- `to_dict()` があればそれを優先
- `__dict__` を持つ場合は `_` で始まる属性を除外
- list/tuple/dict は再帰的に正規化

これにより `__eq__` 実装に依存せず構造比較できます。

### 4.3 2nd pass: pattern_id 付与 & 出力生成
`tmp/_stage_unique.jsonl` を再読込し、以下を生成します。

#### pattern_id の決定
- パース成功時: `id_prefix + ast_sig[:id_hash_len]`
- パース失敗時: `pattern` 文字列のSHA256から同様に生成
- **衝突した場合は `_2`, `_3`... を付与**

#### YAML出力（`patterns/JA_T2KGB/*.yaml`）
最小構成：
- `pattern_id`
- `status`
- `pattern`
- `ast_sig`（存在すれば）
- `parse_error`（存在すれば）

#### index出力
- `index/patterns.jsonl`
  - 1行1パターン。`pattern` 本文と軽量特徴を保持。
- `index/patterns.index.json`
  - `patterns` 配列は **軽量情報のみ**（高速検索向け）。

##### index向け抽出特徴
`pattern` 文字列から以下を抽出します。
- `has_literal`: ダブルクオートで囲まれたリテラルがあるか
- `literals`: `"..."` 内のテキスト（最大50件）
- `pos_constraints`: `-POS` 形式の品詞制約（最大200件）

#### MANIFEST
`index/MANIFEST.json` には以下を記録します。
- 入力ファイルのパスとSHA256
- parser/grammar のSHA256（存在すれば）
- 出力先ディレクトリと件数
- 統計情報（読み込み数、重複数、パース失敗数など）
- 追加フィルタ（リテラル必須・X/Y隣接除外）
- 重複除去ポリシー

### 4.4 最終出力の置換更新
`tmp/` に書いたファイルを `os.replace()` で本番の `index/` に置換します。

## 5. 失敗・除外時の扱い
- JSON不正／pattern欠落／パース失敗（keepなし）
  - `tmp/bad_records.jsonl` に原因と元レコードを保存。
- リテラル無し（`reason=no_literal`）
  - `tmp/bad_records.jsonl` に記録。
- X/Y要素の直接連結（`reason=adjacent_xy`）
  - `tmp/bad_records.jsonl` に記録。
- パース失敗を採用する場合（`--keep-unparsable`）
  - `ast_sig` は `null`、`parse_error` を保持。
- `pattern_parser.py` がロードできない場合
  - 例外で即停止。

## 6. 依存パッケージ
- `tqdm`
- `PyYAML`
  - インストールされていない場合、`yaml_dump()` が例外を投げます。
  - エラーメッセージでは `docker/requirements.txt` に `pyyaml` を追加するよう指示されています。

## 7. 関連プログラム・ディレクトリ
### 直接関連
- `tmp/src/build_repo_from_patterns_from_text.py`（本体）
- `patterns/JA_T2KGB/`（YAML出力）
- `index/`（索引とマニフェスト出力）
- `tmp/`（中間生成物・レポート）
- `../pattern_grammar/`（`pattern_parser.py` と `grammar.lark` を想定）

### 上流（入力生成）
- `tmp/src/build_patterns_from_text.py`
  - `../pattern_candidate/` 以下の `pass_pairs*.jsonl` を集約し `patterns_from_text.jsonl` を生成。
  - 本スクリプトの入力データを作る役割。

### 依存設定
- `docker/requirements.txt`
  - `PyYAML` が未記載のため、YAML出力が必要なら追加が想定されている。

## 8. 注意点・運用上のポイント
- パターンの **重複除去は「文字列一致」と「AST構造一致」** の二段階。
- 採用条件として **「リテラル必須」** と **「X/Y要素の直接連結禁止」** が追加されています。
- `pattern_id` は **ASTに依存した安定ID** のため、入力順序が変わっても一致します。
- `--keep-unparsable` を使うと、**パース不能パターンも採用**されるため、
  `patterns/` や `index/` に `parse_error` を持つレコードが含まれます。
- `index/patterns.jsonl` は `literals` を保持し、`patterns.index.json` は軽量化のため `literals` を持ちません。

## 9. 存在確認できなかったもの
- `generate_pattern_repository.py`
  - 現在の `/workspace` には存在を確認できませんでした。
  - 名前から推測する限り関連しそうですが、本ドキュメントでは参照対象外です。
