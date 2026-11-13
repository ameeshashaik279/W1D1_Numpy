
import chromadb
from langchain_ollama import OllamaEmbeddings

# Local Ollama embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="w6_documents",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "Python is a popular programming language.",
    "Machine learning helps computers learn from data.",
    "Artificial intelligence enables intelligent machines.",
    "Deep learning uses neural networks.",
    "Data science extracts insights from data.",
    "SQL manages relational databases.",
    "Pandas is used for data analysis.",
    "NumPy supports numerical computing.",
    "Matplotlib creates visualizations.",
    "Power BI creates interactive dashboards.",
    "Cloud computing provides internet-based services.",
    "AWS is a cloud computing platform.",
    "Microsoft Azure provides cloud services.",
    "Docker supports containerization.",
    "Git tracks source code changes.",
    "LangChain builds applications using language models.",
    "Ollama runs language models locally.",
    "ChromaDB is a vector database.",
    "Embeddings represent text as numerical vectors.",
    "RAG combines retrieval and generation."
]

ids = [f"doc_{i+1}" for i in range(len(documents))]
metadatas = [
    {"category": "AI" if i < 5 else "Technology"}
    for i in range(len(documents))
]

embeddings = embedding_model.embed_documents(documents)

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Total documents:", collection.count())
print("20 documents added successfully!")
# 1. Cosine Similarity Search
query = "What is machine learning?"

query_embedding = embedding_model.embed_query(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nTOP 3 SIMILAR DOCUMENTS")
for document, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Document: {document}")
    print(f"Cosine distance: {distance:.4f}")
    print("-" * 40)


# 2. Metadata Filtering
filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "AI"}
)

print("\nAI CATEGORY FILTER RESULTS")
for document in filtered_results["documents"][0]:
    print(document)
