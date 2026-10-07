# RAG Research Agent

A Databricks-hosted Python research assistant that extends a simple grounded assistant into a more production-oriented RAG architecture with semantic retrieval, governed model access, observability, and fallback behavior.

This is the second stage of a three-project progression from lexical retrieval to semantic RAG and then multi-agent orchestration.

## What it demonstrates

- Databricks Vector Search for semantic retrieval from a Python knowledge base
- Keyword retrieval fallback when the Vector Search index is unavailable
- AI Gateway / Unity Gateway for governed LLM access
- Foundation Model API fallback when the gateway is unavailable
- MLflow tracing for retriever, LLM, and chain execution
- Unity Catalog inference logging
- Request/response, token usage, model, gateway usage, and latency capture
- Streamlit conversational UI through Databricks Apps
- Multi-turn conversation context

## Architecture

```text
User question
      |
      v
Vector Search
      |
      +---- unavailable/error ----> Keyword fallback
      |
      v
Retrieved Python context
      |
      v
AI Gateway
      |
      +---- unavailable ----------> Foundation Model API
      |
      v
Grounded response
      |
      +----> MLflow trace
      +----> Unity Catalog inference log
```

## Retrieval

The primary retriever queries:

```text
main.default.python_kb_docs_index
```

for semantically relevant Python documents. The application requests the document ID, topic, and content and uses the returned documents as grounding context.

If Vector Search is unavailable, the application falls back to lexical matching against its local Python knowledge base so the assistant can continue operating with reduced retrieval capability.

## Model access

The preferred path uses Databricks AI Gateway / Unity Gateway with:

```text
system.ai.llama-4-maverick
```

The application performs a startup availability check. If the gateway cannot be used, model traffic falls back to:

```text
databricks-meta-llama-3-3-70b-instruct
```

through the Databricks Foundation Model API.

## Observability

MLflow traces the retrieval and generation chain. The application also writes inference metadata to a Unity Catalog Delta table, including the query, retrieved context, response, model, whether the gateway was used, token counts, and latency.

Diagnostic startup logging is best-effort so observability failures do not prevent the research assistant itself from running.

## Project structure

```text
apps/
└── stage2-gateway-vectorsearch/
    ├── app.py
    ├── app.yaml
    ├── requirements.txt
    └── python-3.14.8-docs.epub
```

The current implementation is kept in a single application file because this project captures the second-stage experiment. The next project refactors responsibilities into explicit agent boundaries.

## Project progression

**Stage 1 — [Research AI Assistant](https://github.com/AnaghaDhekne/research-ai-assistant)**  
Establishes the grounded-generation baseline using lexical retrieval, a Databricks Foundation Model, Streamlit, and MLflow tracing.

**Stage 2 — RAG Research Agent (this repository)**  
Adds semantic Vector Search, retrieval fallback, governed model access, inference logging, and richer observability.

**Stage 3 — [Multi-Agent Research System](https://github.com/AnaghaDhekne/multi-agent-research-system)**  
Reuses the Python semantic-retrieval capability as a specialist tool and adds routing between Python and Databricks domains, parallel specialist execution, and evidence-grounded synthesis.

## Current scope

This repository demonstrates retrieval and model-serving infrastructure rather than autonomous agents. Its `chat_agent` function is an orchestrated RAG chain; the explicit multi-agent architecture is introduced only in Stage 3.
