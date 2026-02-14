#!/usr/bin/env python3
"""Retrieve top-k chunks from OpenSearch Serverless using a user query.

Flow:
1) Read query (CLI arg or stdin)
2) Embed query with Bedrock Titan Text Embeddings V2
3) kNN search in OpenSearch index (rag-chunks-v1)

Required env vars (in .env or environment):
- OPENSEARCH_END_POINT=https://...aoss.amazonaws.com
- INDEX_NAME=rag-chunks-v1
Optional:
- AWS_REGION=us-east-1
- BEDROCK_MODEL_ID=amazon.titan-embed-text-v2:0
"""

import json
import os
import sys
from typing import Any, Dict, List

import boto3
import botocore
import requests
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing {name} env var")
    return val


def make_awsauth(region: str, service: str = "aoss") -> AWS4Auth:
    """Create SigV4 auth for OpenSearch Serverless (service = aoss)."""
    session = boto3.Session(region_name=region)
    creds = session.get_credentials()
    if creds is None:
        raise SystemExit("No AWS credentials found. Configure AWS_PROFILE or AWS_ACCESS_KEY_ID/...")
    frozen = creds.get_frozen_credentials()
    return AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        service,
        session_token=frozen.token,
    )


def bedrock_embed(text: str, region: str, model_id: str) -> List[float]:
    """Embed text with Amazon Titan Text Embeddings.

    Different Titan embedding models/versions accept slightly different request shapes.
    We try the v2-style shape first, then fall back.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    # Try Titan Text Embeddings V2 request shape first
    payload_v2 = {
        "inputText": text,
        "embeddingConfig": {"outputEmbeddingLength": 1024},
    }

    # Fallback: some models only accept inputText (no configurable length)
    payload_simple = {"inputText": text}

    def _invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        return json.loads(resp["body"].read())

    try:
        body = _invoke(payload_v2)
    except botocore.exceptions.ClientError as e:
        # If the model rejects embeddingConfig / outputEmbeddingLength, retry without it.
        msg = str(e)
        if "extraneous key" in msg or "embeddingConfig" in msg or "outputEmbeddingLength" in msg:
            body = _invoke(payload_simple)
        else:
            raise

    # Common response: {"embedding": [...]}
    if "embedding" in body and isinstance(body["embedding"], list):
        return body["embedding"]

    # Fallbacks just in case response shape differs
    for k in ("embeddings", "vector"):
        if k in body and isinstance(body[k], list):
            return body[k]

    raise RuntimeError(f"Unexpected Bedrock response keys: {list(body.keys())}")


def opensearch_knn_search(
    endpoint: str,
    index: str,
    region: str,
    query_vector: List[float],
    k: int = 5,
) -> Dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{index}/_search"

    # OpenSearch Serverless uses service name 'aoss' for SigV4
    awsauth = make_awsauth(region=region, service="aoss")

    query = {
        "size": k,
        "_source": ["chunk_id", "document_id", "page_num", "chunk_text"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    r = requests.post(url, auth=awsauth, json=query, timeout=60)
    if r.status_code >= 300:
        # print raw to help debug 4xx/5xx
        raise RuntimeError(f"OpenSearch error {r.status_code}: {r.text}")
    return r.json()


def read_query_from_cli() -> str:
    if len(sys.argv) >= 2:
        # allow: python 04_retrieve_topk.py "my question"
        return " ".join(sys.argv[1:]).strip()

    # stdin fallback
    print("Enter your question:")
    q = sys.stdin.readline().strip()
    return q


def main() -> None:
    load_dotenv()

    endpoint = require_env("OPENSEARCH_END_POINT")
    index = require_env("INDEX_NAME")

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")

    question = read_query_from_cli()
    if not question:
        raise SystemExit("Empty query")

    print(f"Query: {question}")
    qvec = bedrock_embed(question, region=region, model_id=model_id)
    if len(qvec) != 1024:
        raise SystemExit(f"Query embedding dimension {len(qvec)} != 1024 (mapping expects 1024)")

    result = opensearch_knn_search(
        endpoint=endpoint,
        index=index,
        region=region,
        query_vector=qvec,
        k=5,
    )

    hits = result.get("hits", {}).get("hits", [])
    print(f"\nTop {len(hits)} results:\n")

    for i, h in enumerate(hits, 1):
        src = h.get("_source", {})
        score = h.get("_score")
        chunk_id = src.get("chunk_id")
        page_num = src.get("page_num")
        text = src.get("chunk_text", "")

        preview = text.replace("\n", " ").strip()
        if len(preview) > 260:
            preview = preview[:260] + "..."

        print(f"{i}. score={score} page={page_num} chunk_id={chunk_id}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()