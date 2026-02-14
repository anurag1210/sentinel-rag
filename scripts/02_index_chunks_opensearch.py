#Script to generate the embeddings at run time
#Importing the necessary packages
import json
import os
import boto3
import requests
from requests_aws4auth import AWS4Auth
from pathlib import Path
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]

model_id = "amazon.titan-embed-text-v2:0"


def get_bedrock_embedding(text: str, region: str) -> list[float]:

    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True,
    }
    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    payload = json.loads(resp["body"].read())

    return payload["embedding"]


#Reading the first chunk

def iter_chunks(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def make_awsauth(region: str, service: str = "aoss") -> AWS4Auth:
    """Creates SigV4 auth for OpenSearch Serverless (service = aoss)."""
    session = boto3.Session()
    creds = session.get_credentials()
    frozen = creds.get_frozen_credentials()
    return AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        service,
        session_token=frozen.token,
    )


def chunk_exists(endpoint: str, region: str, index_name: str, chunk_id: str) -> bool:
    """Return True if a document with this chunk_id already exists in the index."""
    auth = make_awsauth(region=region, service="aoss")
    headers = {"Content-Type": "application/json"}

    search_url = f"{endpoint.rstrip('/')}/{index_name}/_search"
    query = {
        "size": 0,
        "query": {
            "term": {
                "chunk_id": chunk_id
            }
        }
    }

    resp = requests.post(search_url, auth=auth, headers=headers, data=json.dumps(query))
    if resp.status_code != 200:
        raise SystemExit(f"Failed to search for {chunk_id}: {resp.status_code} {resp.text}")

    data = resp.json()
    total = data.get("hits", {}).get("total", {})
    # OpenSearch can return total as an int or an object
    if isinstance(total, int):
        return total > 0
    return total.get("value", 0) > 0


def index_one_doc(endpoint: str, region: str, index_name: str, doc: dict) -> requests.Response:
    """POST a document into OpenSearch Serverless. (Do NOT set _id; AOSS can reject custom IDs.)"""
    auth = make_awsauth(region=region, service="aoss")
    headers = {"Content-Type": "application/json"}

    url = f"{endpoint.rstrip('/')}/{index_name}/_doc"
    return requests.post(url, auth=auth, headers=headers, data=json.dumps(doc, ensure_ascii=False))



def main():
    #Load the .env environment variables.
    load_dotenv()
    endpoint = os.environ.get("OPENSEARCH_END_POINT")
    index_name=os.environ.get("INDEX_NAME")
    input_jsonl=os.environ.get("INPUT_JSONL")
    if not endpoint:
        raise SystemExit("Missing OPENSEARCH_END_POINT env var (e.g. https://xxxx.us-east-1.aoss.amazonaws.com)")
    if not index_name:
        raise SystemExit("Missing INDEX_NAME env var (e.g. rag-chunks-v1)")
    if not input_jsonl:
        raise SystemExit("Missing INPUT_JSONL env var (e.g. data/processed/output/combined.jsonl)")

    region = os.environ.get("AWS_REGION", "us-east-1")



    input_path = Path(input_jsonl)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path


    # Debug: print index URL once
    print("Index URL:", f"{endpoint.rstrip('/')}/{index_name}/_doc")

    total = 0
    success = 0

    for chunk in iter_chunks(str(input_path)):
        total += 1

        chunk_id = chunk["chunk_id"]

        # 0) Skip if already indexed (saves Bedrock embedding cost)
        if chunk_exists(endpoint, region, index_name, chunk_id):
            print(f"Skip (already indexed): {chunk_id}")
            continue

        # 1) Embed chunk_text
        embedding = get_bedrock_embedding(chunk["chunk_text"], region=region)

        # 2) Build OpenSearch doc (field names must match your mapping)
        os_doc = {
            "chunk_id": chunk_id,
            "document_id": chunk["document_id"],
            "page_num": int(chunk["page_num"]),
            "chunk_text": chunk["chunk_text"],
            "embedding": embedding,  # must be length 512
        }

        assert len(os_doc["embedding"]) == 1024, f"Expected 1024-dim embedding, got {len(os_doc['embedding'])}"

        # 3) Index it
        resp = index_one_doc(endpoint, region, index_name, os_doc)

        if resp.status_code in (200, 201):
            success += 1
            print(f"Indexed {success}/{total}: {os_doc['chunk_id']}")
        else:
            print("Indexing failed for chunk_id:", os_doc["chunk_id"])
            print("Status:", resp.status_code)
            try:
                print("Body:", resp.json())
            except Exception:
                print("Body (raw):", resp.text)
            raise SystemExit("Indexing failed — check the response above.")

    print(f"Done. Indexed {success} documents.")


if __name__ == "__main__":
    main()