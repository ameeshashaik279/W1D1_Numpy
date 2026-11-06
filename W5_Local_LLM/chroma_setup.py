
import chromadb
import requests

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="week5_documents",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "Python is a popular programming language.",
    "Machine learning helps computers learn from data.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Data science combines statistics, programming, and domain knowledge.",
    "SQL is used to manage relational databases.",
    "Pandas is a Python library for data analysis.",
    "NumPy supports numerical computing in Python.",
    "Matplotlib is used for data visualization.",
    "Deep learning uses neural networks.",
    "Natural language processing works with human language.",
    "Computer vision helps computers understand images.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "Classification predicts categories.",
    "Regression predicts numerical values.",
    "Clustering groups similar data points.",
    "Feature engineering improves machine learning inputs.",
    "Model evaluation measures prediction performance.",
    "ChromaDB stores and searches vector embeddings.",
    "Ollama runs large language models locally."
]

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]

for i, document in enumerate(documents):
    embedding = get_embedding(document)

    collection.upsert(
        ids=[str(i)],
        documents=[document],
        embeddings=[embedding],
        metadatas=[{"category": "AI_ML", "index": i}]
    )

print("20 documents added successfully!")

query = "What is machine learning?"
query_embedding = get_embedding(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nTop 3 similar documents:")
for document in results["documents"][0]:
    print("-", document)

print("\nMetadata filtering:")
filtered = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"category": "AI_ML"}
)

for document in filtered["documents"][0]:
    print("-", document)
