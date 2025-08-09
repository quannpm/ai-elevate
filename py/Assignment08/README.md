# Semantic Search Engine for Clothing Products

## Description

This script builds a semantic search engine for clothing products using embeddings from Azure OpenAI (`text-embedding-3-small`). Users enter a query (desired product description), and the system returns the most relevant products based on semantic similarity.

---

## How It Works

- The script generates embeddings for each product description and for the user's query using Azure OpenAI.
- Cosine similarity is used to measure how close the query is to each product in semantic space.
- The top matching products are returned and displayed.

---

## Cosine Similarity Explained

- Cosine similarity measures the angle between two vectors (embeddings).
- A value close to 1 means the two descriptions are semantically very similar.

---

## Usage Instructions

1. **Install required libraries:**
    ```
    pip install openai scipy
    ```

2. **Set Azure OpenAI environment variables:**
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_API_KEY`

3. **Run the script:**
    ```
    python 8.py
    ```

4. **Result:**  
    The script prints the top 3 most relevant products for your query, along with their similarity scores.

---

## Limitations & Challenges

- Requires a valid Azure OpenAI API key and endpoint.
- If product descriptions or queries are too short, embeddings may not fully capture their meaning.
- For large product datasets, embedding generation via API may be slow.

---

## Example Product Data

```python
products = [
    {
        "title": "Classic Blue Jeans",
        "short_description": "Comfortable blue denim jeans with a relaxed fit.",
        "price": 49.99,
        "category": "Jeans"
    },
    {
        "title": "Red Hoodie",
        "short_description": "Cozy red hoodie made from organic cotton.",
        "price": 39.99,
        "category": "Hoodies"
    },
    # ...
]
```