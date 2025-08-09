# Semantic Search Engine for Clothing Products
# ----------------------------------------------------------
# How to run:
# 1. Install required libraries:
#    pip install openai scipy
#
# 2. Set Azure OpenAI environment variables before running:
#    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
#    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
#    Example (Windows):
#      set AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
#      set AZURE_OPENAI_API_KEY=<your-api-key>
#
# You can change the 'query' variable to test different search terms.
# Add more products to the 'products' list as needed.
# ----------------------------------------------------------

import os
from openai import AzureOpenAI 
from scipy.spatial.distance import cosine 

# Step 1: Setup AzureOpenAI  
client = AzureOpenAI(  
    api_version="2024-07-01-preview",  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
) 

# Step 2: Sample product data 
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
    { 
        "title": "Black Leather Jacket", 
        "short_description": "Stylish black leather jacket with a slim fit design.", 
        "price": 120.00, 
        "category": "Jackets" 
    }, 
    # Add more products as needed 
] 

# Step 3: Function to get embeddings from Azure OpenAI 
def get_embedding(text): 
    response = client.embeddings.create( 
        model="text-embedding-3-small", 
        input=text 
    ) 
    embedding = response.data[0].embedding 
    return embedding 

# Step 4: Generate embeddings for all product descriptions 
for product in products: 
    product["embedding"] = get_embedding(product["short_description"]) 

# Step 5: Accept user input (query) 
query = "warm cotton sweatshirt"  # You can change this query to test other searches

# Step 6: Get embedding for the user query 
query_embedding = get_embedding(query) 

# Step 7: Compute cosine similarity between query and each product 
def similarity_score(vec1, vec2): 
    return 1 - cosine(vec1, vec2)  # cosine returns distance; 1 - distance = similarity 

scores = [] 
for product in products: 
    score = similarity_score(query_embedding, product["embedding"]) 
    scores.append((score, product)) 

# Step 8: Sort products by similarity descending 
scores.sort(key=lambda x: x[0], reverse=True) 

# Step 9: Display top matches 
print(f"Top matching products for query: '{query}'\n") 
for score, product in scores[:3]:  # top 3 results 
    print(f"Title: {product['title']}") 
    print(f"Description: {product['short_description']}") 
    print(f"Price: ${product['price']}") 
    print(f"Category: {product['category']}") 
    print(f"Similarity Score: {score:.4f}\n")