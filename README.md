Sentinel-RAG

Production-Style Retrieval-Augmented Generation System

🚀 Overview

Sentinel-RAG is a production-oriented Retrieval-Augmented Generation (RAG) system built using:
	•	AWS Bedrock (Claude)
	•	OpenSearch (vector database)
	•	FastAPI (backend API)
	•	Next.js (frontend UI)
	•	Server-Sent Events (streaming inference)

The system enables document-grounded LLM responses with real-time token streaming and citation tracing.

⸻

🧠 Problem It Solves

Large Language Models hallucinate when answering without grounding.

Sentinel-RAG solves this by:
	•	Embedding user queries
	•	Performing semantic similarity search
	•	Retrieving relevant document chunks
	•	Injecting context into the LLM prompt
	•	Streaming grounded responses back to the user

⸻

🏗 Architecture


Flow:
	1.	User sends question via Next.js UI
	2.	API Route proxies to FastAPI backend
	3.	Query embedding generated (Bedrock Titan)
	4.	Vector similarity search in OpenSearch
	5.	Top-K chunks injected into Claude prompt
	6.	Claude response streamed via SSE
	7.	Frontend renders tokens in real time

⸻

⚙️ Key Engineering Features
	•	Embedding-based semantic retrieval (1024-dim vectors)
	•	Vector search using OpenSearch Serverless
	•	RESTful API design (FastAPI)
	•	Server-Sent Events (SSE) streaming
	•	Prompt grounding to reduce hallucination
	•	Metadata tracing (page number + chunk_id)
	•	Latency tracking and cost-awareness considerations
	•	Clean separation of backend and BFF layer

⸻

📊 Evaluation Considerations
	•	Retrieval precision depends on chunk size & embedding quality
	•	Latency primarily driven by LLM inference time
	•	Streaming improves perceived latency
	•	Grounding reduces hallucination risk

⸻

🔐 Safety & Guardrails
	•	Context-only answering enforced in prompt
	•	Controlled output formatting
	•	Designed to reject answers outside provided document context

⸻

🛠 Tech Stack

Backend:
	•	Python
	•	FastAPI (async)
	•	boto3 (Bedrock integration)
	•	OpenSearch
	•	AWS4Auth

Frontend:
	•	Next.js
	•	TypeScript
	•	Streaming via ReadableStream

Cloud:
	•	AWS Bedrock
	•	OpenSearch Serverless

⸻

📌 Lessons Learned
	•	LLM inference dominates latency
	•	Streaming improves UX but not total compute time
	•	Vector dimensional consistency is critical
	•	Proper environment configuration avoids subtle failures
	•	Prompt structure directly impacts hallucination behavior
