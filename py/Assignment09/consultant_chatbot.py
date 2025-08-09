import os
import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
# This allows us to store sensitive information (API keys, endpoints) in a separate .env file
load_dotenv()

# ====== Azure OpenAI Embedding Client ======
# Create a client for generating embeddings using Azure OpenAI
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),  # Load embedding API key from environment
    api_version="2024-02-15-preview",  # API version for Azure OpenAI
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")  # Load embedding endpoint
)
# Load embedding model name from environment or fallback to a default
embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ====== Azure OpenAI LLM Client ======
# Create a client for Large Language Model (LLM) interactions
llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),  # Load LLM API key from environment
    api_version="2024-02-15-preview",  # API version for LLM
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT")  # Load LLM endpoint
)
# Load LLM model name from environment or fallback to a default
llm_model = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")

# ====== ChromaDB Persistent Client ======
# Initialize ChromaDB client with persistence (data stored on disk)
chroma_client = chromadb.PersistentClient(path="./chroma_store")
# Create or get an existing collection for storing laptop recommendation documents
collection = chroma_client.get_or_create_collection(name="laptop_recommendations")


def embed_text(text: str):
    """Generate embeddings for a given text."""
    # Call Azure OpenAI embedding API to convert text into vector representation
    response = embedding_client.embeddings.create(
        model=embed_model,
        input=text
    )
    # Return the embedding vector for the first (and only) input text
    return response.data[0].embedding


def add_document(doc_id: str, content: str, metadata: dict):
    """Add a document to ChromaDB."""
    # Generate embedding for the given document content
    embedding = embed_text(content)
    # Store the document, metadata, and embedding in the ChromaDB collection
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata],
        embeddings=[embedding]
    )


def build_context(query: str, n_context: int = 3):
    """Retrieve top matching documents from ChromaDB."""
    # Generate embedding for the user's query
    query_embedding = embed_text(query)
    # Query ChromaDB for the most relevant documents based on cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_context
    )
    # Extract document texts from the query results
    documents = results.get("documents", [[]])[0]
    # Join multiple documents into a single context string
    return "\n".join(documents)


def chat_with_rag(user_query: str):
    """Chat with LLM using RAG (context + user query)."""
    # Step 1: Retrieve relevant documents for the query
    context = build_context(user_query, n_context=3)
    # Step 2: Create a system prompt to instruct the LLM
    system_prompt = (
        "You are a helpful assistant specializing in laptop recommendations. "
        "Answer based on the provided context and your own knowledge."
    )
    
    # Step 3: Prepare the conversation messages
    messages = [
        {"role": "system", "content": system_prompt},  # System role sets assistant's behavior
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}  # User's query with retrieved context
    ]
    
    # Step 4: Send the request to Azure OpenAI LLM
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.7  # Controls randomness in the response
    )
    
    # Step 5: Return the LLM's answer
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Starting...")

    add_document(
        doc_id="doc1",
        content="The Dell XPS 13 has a 13-inch display, Intel i7 processor, and 16GB RAM.",
        metadata={"brand": "Dell", "category": "Ultrabook"}
    )
    print("Added sample 1")
    
    add_document(
        doc_id="doc2",
        content="The MacBook Air M2 offers excellent battery life and performance for daily use.",
        metadata={"brand": "Apple", "category": "Laptop"}
    )
    print("Added sample 2")

    user_question = "Which laptop is best for programming?"
    print(f"User question: {user_question}")
    
    answer = chat_with_rag(user_question)
    print("💬 Answer:", answer)
