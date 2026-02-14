import boto3
import json
from pathlib import Path

INPUT_PATH = Path("/Users/anuraggupta/projects/sentinel-rag/data/processed/output/combined.jsonl")

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-embed-text-v2:0"

def embed_text(text: str) -> list[float]:
    request = {
        "inputText": text,
        # Titan Text Embeddings v2 defaults to 1024 dims if not specified.
        # Force 512 dims to match the OpenSearch vector index you created.
        "dimensions": 512,
        # Normalize vectors so dot-product search behaves like cosine similarity.
        "normalize": True,
    }
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request),
        contentType="application/json",
        accept="application/json",
    )
    model_response = json.loads(response["body"].read())
    return model_response["embedding"]

def process_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk_text = obj.get("chunk_text")
            if not chunk_text:
                print(f"Skipping line {line_num}: missing chunk_text")
                continue

            embedding = embed_text(chunk_text)
            if len(embedding) != 512:
                raise ValueError(f"Unexpected embedding dimension {len(embedding)} on line {line_num}; expected 768")
            print(f"Line {line_num}: embedding length {len(embedding)}")
            print(f"Line {line_num}: embedding values: {embedding[:10]}")
           

process_jsonl(INPUT_PATH)
