import json
import os
from typing import List, Dict, Any

import boto3
import botocore
import requests
from requests_aws4auth import AWS4Auth

from dotenv import load_dotenv
load_dotenv()
import time
from typing import Generator

# ---------- ENV HELPERS ----------

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing {name} environment variable")
    return val


def _make_awsauth(region: str, service: str = "aoss") -> AWS4Auth:
    session = boto3.Session(region_name=region)
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError("No AWS credentials found")

    frozen = creds.get_frozen_credentials()

    return AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        service,
        session_token=frozen.token,
    )


# ---------- EMBEDDING ----------


def embed_text(text: str) -> List[float]:
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")

    client = boto3.client("bedrock-runtime", region_name=region)

    payload = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True,
    }

    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    body = json.loads(resp["body"].read())

    if "embedding" not in body:
        raise RuntimeError(f"Unexpected embedding response: {body}")

    emb = body["embedding"]
    if not isinstance(emb, list):
        raise RuntimeError(f"Embedding is not a list: {type(emb)}")
    if len(emb) != 1024:
        raise RuntimeError(f"Embedding dimension {len(emb)} != 1024 (expected 1024)")

    return emb


# ---------- RETRIEVAL ----------

def retrieve_chunks(query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    endpoint = _require_env("OPENSEARCH_END_POINT")
    index = _require_env("INDEX_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")

    url = f"{endpoint.rstrip('/')}/{index}/_search"

    awsauth = _make_awsauth(region)

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
        raise RuntimeError(f"OpenSearch error {r.status_code}: {r.text}")

    result = r.json()

    return result.get("hits", {}).get("hits", [])


# ---------- GENERATION (CLAUDE) ----------

def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv(
        "BEDROCK_LLM_MODEL_ID",
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    context_parts = []
    for h in chunks:
        src = h.get("_source", {})
        page = src.get("page_num")
        chunk_id = src.get("chunk_id")
        text = src.get("chunk_text", "")
        context_parts.append(f"(p.{page}, {chunk_id})\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant.
Answer the QUESTION using ONLY the CONTEXT below.
If the answer is not in the context, say exactly:
I can’t find that in the provided document.

Return your answer as bullet points.
Use ONLY the citations that appear in the CONTEXT headers (the ones like (p.X, chunk_id)).
Do not invent page numbers or chunk_ids.
After EACH bullet, add citations in this exact format: (p.<page>, <chunk_id>)

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    response_body = json.loads(resp["body"].read())

    return response_body["content"][0]["text"]




# ---------- FULL RAG PIPELINE ----------

def ask_rag(question: str) -> Dict[str, Any]:
    t0 = time.time()

    query_vector = embed_text(question)
    chunks = retrieve_chunks(query_vector, k=5)
    answer = generate_answer(question, chunks)

    latency_ms = int((time.time() - t0) * 1000)

    # Build sources from retrieved chunks
    sources = []
    for h in chunks:
        src = h.get("_source", {})
        sources.append({
            "chunk_id": src.get("chunk_id"),
            "page": src.get("page_num"),
        })

    return {
        "answer": answer,
        "sources": sources,
        "metadata": {
            "model": os.getenv(
                "BEDROCK_LLM_MODEL_ID",
                "anthropic.claude-3-sonnet-20240229-v1:0"
            ),
            "chunks_used": len(chunks),
            "latency_ms": latency_ms,
        }
    }


#----------Ask RAG for streaming response----------


def ask_rag_stream(question: str) -> Generator[str, None, None]:
    start_time = time.time()

    # 1) Embed
    query_vector = embed_text(question)

    # 2) Retrieve
    chunks = retrieve_chunks(query_vector, k=5)

    # Build context
    context_parts = []
    for h in chunks:
        src = h.get("_source", {})
        page = src.get("page_num")
        chunk_id = src.get("chunk_id")
        text = src.get("chunk_text", "")
        context_parts.append(f"(p.{page}, {chunk_id})\n{text}")

    context = "\n\n".join(context_parts)

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

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv(
        "BEDROCK_LLM_MODEL_ID",
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    #Method which invokes model to stream with a response stream 
    response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    # Stream tokens
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])

        if chunk["type"] == "content_block_delta":
            delta = chunk["delta"].get("text")
            if delta:
                yield delta

    # Final metadata
    latency_ms = int((time.time() - start_time) * 1000)

    yield "\n\n[END]"