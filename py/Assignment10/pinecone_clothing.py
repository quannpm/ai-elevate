"""
Semantic Vector Similarity with Pinecone (Sample Clothing Products)
Install:
   pip install pinecone python-dotenv
Run:
   python pinecone_clothing.py
Environment:
   set PINECONE_API_KEY=YOUR_KEY
"""

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing PINECONE_API_KEY.")

INDEX_NAME = "product-similarity-index"
DIMENSION = 4
METRIC = "cosine"
TOP_K = 3

products: List[Dict[str, Any]] = [
    {"id": "prod1", "title": "Red T-Shirt", "description": "Comfortable cotton t-shirt in bright red", "embedding": [0.12, 0.98, 0.34, 0.56]},
    {"id": "prod2", "title": "Blue Jeans", "description": "Stylish denim jeans with relaxed fit", "embedding": [0.10, 0.88, 0.40, 0.60]},
    {"id": "prod3", "title": "Black Leather Jacket", "description": "Genuine leather jacket with classic style", "embedding": [0.90, 0.12, 0.75, 0.15]},
    {"id": "prod4", "title": "White Sneakers", "description": "Comfortable sneakers perfect for daily wear", "embedding": [0.20, 0.95, 0.38, 0.55]},
    {"id": "prod5", "title": "Green Hoodie", "description": "Warm hoodie made of organic cotton", "embedding": [0.15, 0.93, 0.35, 0.50]},
]

query_embedding = [0.18, 0.90, 0.40, 0.52]


def ensure_index(pc: Pinecone, index_name: str):
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"[INFO] Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while True:
            status = pc.describe_index(index_name).status
            if status.get("ready"):
                break
            print("  - Waiting for index to be ready...")
            time.sleep(1)
        print("[INFO] Index created.")
    else:
        print(f"[INFO] Index '{index_name}' already exists.")


def build_vectors(products: List[Dict[str, Any]]):
    return [
        {
            "id": p["id"],
            "values": p["embedding"],
            "metadata": {"title": p["title"], "description": p["description"]}
        }
        for p in products
    ]


def main():
    pc = Pinecone(api_key=API_KEY)
    ensure_index(pc, INDEX_NAME)
    index = pc.Index(INDEX_NAME)

    vectors = build_vectors(products)
    index.upsert(vectors)
    print(f"[INFO] Upserted {len(vectors)} product vectors.")

    print(f"\n[INFO] Querying top {TOP_K} similar products...\n")
    result = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_values=False,
        include_metadata=True
    )

    print("Top results:")
    for match in result.matches:
        meta = match.metadata or {}
        print(f"- {meta.get('title')} | Score: {match.score:.4f}")
        print(f"  Description: {meta.get('description')}")
    print("\n[DONE] Similarity search completed.")


if __name__ == "__main__":
    main()