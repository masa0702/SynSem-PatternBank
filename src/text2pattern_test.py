# run_one.py
import json

from pattern_repo_client import (
    load_json_file,
    OllamaConfig,
    OllamaJSONSchemaClient,
    PatternGenerator,
)

def main():
    schema = load_json_file("../schema/schema.json")  # PatternRepoOutput 用
    
    cfg = OllamaConfig(
        base_url="http://ollama:11434",
        model="gpt-oss:20b",
        endpoint="chat",
        temperature=0.0,
        seed=42,
        max_tokens=4048,  # patternが短いなら 64 まで落としてOK
        timeout_sec=300,
        max_retries=3,
    )

    client = OllamaJSONSchemaClient(cfg, schema)
    gen = PatternGenerator(client)
    # gen = PatternGenerator(
    #     client,
    #     enable_bunsetsu=True,
    #     bunsetsu_module_path="./bunsetu.py",
    # )


    sentence = "カッサームロケットは、カッサム旅団によってデザインされた、簡素な鋼鉄製の大砲ロケットである。"
    triples = [
        (0, ["カッサームロケット", "デザイン", "カッサム旅団"]),
    ]

    result = gen.generate_one(sentence, triples, debug_dump_path="debug_last.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    output_json_path = "../output/output.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    main()
