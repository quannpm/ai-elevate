# Retail RAG Chatbot

This package contains an implementation of a Retrieval-Augmented Generation (RAG) chatbot for retail product and policy support.

## Files produced

- `rag.py` — The main script (this file).
- `demo_results.json` — Demo outputs for sample questions.

## How to run

1. Install required packages (example):

```bash
pip3 install langchain langchain-community langchain-core langgraph faiss-cpu openai
```

2. Set environment variables (optional, if you want real OpenAI calls):

- `OPENAI_API_KEY`

3. Run:

```
python3 rag.py
```

## Final submission checklist

- [x] Source code implementing the RAG chatbot (this .py file).
- [x] At least one realistic Q&A demo showing input question, retrieved context, and generated answer (demo_results.json).
- [x] Dummy input list used (no manual input).
- [x] Brief reflection on how RAG improves retail support vs. static FAQ (included below).
- [x] Optional suggestions for expanding to product recommendations or multi-turn conversations (included below).

## Reflection

Reflection:
RAG (Retrieval-Augmented Generation) improves retail support by combining an up-to-date knowledge base with a generative model. Instead of relying on static FAQ pages or keyword search, RAG retrieves the most relevant passages and conditions the LLM's answer on them, producing responses that are grounded in documented policy while still being conversational and adaptable to the user's exact question. This reduces incorrect answers, shortens support time, and gives staff and customers fast access to policy details.

## Suggestions

Suggestions for future expansion:
1) Store documents in a persistent vector DB (e.g., Pinecone, Milvus) and add incremental updates for new policies.
2) Add multi-turn memory and context tracking for follow-up questions.
3) Integrate product SKU metadata to do hybrid RAG+filtering for product-specific queries and recommendations.
4) Add user authentication and role-based responses for staff vs customers.

