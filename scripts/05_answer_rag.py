#!/usr/bin/env python3
"""
05_answer_rag.py
Full RAG pipeline:
1) Read user question
2) Embed question with Titan Embeddings v2
3) kNN search OpenSearch Serverless for top-k chunks
4) Send retrieved chunks + question to Claude Sonnet
5) Print answer

Env vars (.env):
- OPENSEARCH_END_POINT=https://...aoss.amazonaws.com
- INDEX_NAME=rag-chunks-v1
Optional:
- AWS_REGION=us-east-1
- BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
- BEDROCK_LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
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


# ----------------- Utilities -----------------

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


def read_query() -> str:
    if len(sys.argv) >= 2:
        return " ".join(sys.argv[1:]).strip()
    print("Enter your question:")
    return sys.stdin.readline().strip()


# ----------------- Embeddings (Titan v2) -----------------

def bedrock_embed(text: str, region: str, model_id: str, expected_dim: int = 1024) -> List[float]:
    """
    Embed text with Titan Text Embeddings v2.
    Your index mapping expects 1024 dims (based on your last fixes).
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    payload_v2 = {
        "inputText": text,
        "embeddingConfig": {"outputEmbeddingLength": expected_dim},
    }
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
        msg = str(e)
        # if model rejects embeddingConfig/outputEmbeddingLength, retry simple
        if "extraneous key" in msg or "embeddingConfig" in msg or "outputEmbeddingLength" in msg:
            body = _invoke(payload_simple)
        else:
            raise

    if "embedding" in body and isinstance(body["embedding"], list):
        vec = body["embedding"]
    elif "vector" in body and isinstance(body["vector"], list):
        vec = body["vector"]
    elif "embeddings" in body and isinstance(body["embeddings"], list):
        vec = body["embeddings"]
    else:
        raise RuntimeError(f"Unexpected Bedrock response keys: {list(body.keys())}")

    if len(vec) != expected_dim:
        raise SystemExit(f"Query embedding dimension {len(vec)} != {expected_dim} (mapping expects {expected_dim})")

    return vec


# ----------------- Retrieval (OpenSearch kNN) -----------------

def opensearch_knn_search(
    endpoint: str,
    index: str,
    region: str,
    query_vector: List[float],
    k: int = 5,
) -> Dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{index}/_search"
    awsauth = make_awsauth(region=region, service="aoss")

    query = {
        "size": k,
        "_source": ["chunk_id", "document_id", "page_num", "chunk_text"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k
                }
            }
        },
    }

    r = requests.post(url, auth=awsauth, json=query, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"OpenSearch error {r.status_code}: {r.text}")
    return r.json()


# ----------------- Generation (Claude Sonnet) -----------------

def claude_answer(question: str, chunks: List[Dict[str, Any]], region: str, model_id: str) -> str:
    """
    Uses Claude Sonnet on Bedrock to answer using retrieved context.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    # Keep context compact to avoid huge prompts
    context_lines = []
    for c in chunks:
        page = c.get("page_num")
        chunk_id = c.get("chunk_id")
        text = (c.get("chunk_text") or "").strip()
        context_lines.append(f"[page={page} chunk_id={chunk_id}]\n{text}")

    context = "\n\n---\n\n".join(context_lines)

    prompt = f"""
                You are a helpful assistant.
                Answer the QUESTION using ONLY the CONTEXT below.
                If the answer is not in the context, say exactly:
                I can’t find that in the provided document.

                Return your answer as bullet points.
                After EACH bullet, add citations in this exact format: (p.<page>, <chunk_id>)

                CONTEXT:
                {context}

                QUESTION:
                {question}

                ANSWER:
                """

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    out = json.loads(resp["body"].read())

    # Claude responses: {"content":[{"type":"text","text":"..."}], ...}
    return out["content"][0]["text"]


# ----------------- Main -----------------

def main() -> None:
    load_dotenv()

    endpoint = require_env("OPENSEARCH_END_POINT")
    index = require_env("INDEX_NAME")

    region = os.getenv("AWS_REGION", "us-east-1")

    embed_model_id = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    llm_model_id = os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    question = read_query()
    if not question:
        raise SystemExit("Empty question")

    print(f"\nQuestion: {question}")

    print("\n1) Embedding question...")
    qvec = bedrock_embed(question, region=region, model_id=embed_model_id, expected_dim=1024)

    print("2) Retrieving top chunks from OpenSearch...")
    res = opensearch_knn_search(endpoint, index, region, qvec, k=5)

    hits = res.get("hits", {}).get("hits", [])
    chunks = [h.get("_source", {}) for h in hits]

    print("\nTop chunks:")
    for i, c in enumerate(chunks, 1):
        preview = (c.get("chunk_text") or "").replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"{i}. page={c.get('page_num')} chunk_id={c.get('chunk_id')}  -> {preview}")

    print("\n3) Asking Claude Sonnet...")
    answer = claude_answer(question, chunks, region=region, model_id=llm_model_id)

    print("\n================= ANSWER =================\n")
    print(answer)
    print("\n=========================================\n")


if __name__ == "__main__":
    main()