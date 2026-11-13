
import requests
import chromadb
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# 1. Read PDF
reader = PdfReader("sample.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# 2. Split text into chunks
words = text.split()
chunks = [" ".join(words[i:i + 100]) for i in range(0, len(words), 100)]

# 3. Create ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="pdf_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# 4. Generate embeddings and store chunks
for i, chunk in enumerate(chunks):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": chunk}
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]

    collection.upsert(
        ids=[f"pdf_chunk_{i}"],
        documents=[chunk],
        embeddings=[embedding]
    )

print(f"Successfully stored {len(chunks)} PDF chunks!")

# 5. Ask a question
question = input("\nAsk a question about the PDF: ")

response = requests.post(
    f"{OLLAMA_URL}/api/embeddings",
    json={"model": EMBED_MODEL, "prompt": question}
)
response.raise_for_status()
query_embedding = response.json()["embedding"]

# 6. Retrieve top 3 relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, len(chunks))
)

context = "\n".join(results["documents"][0])

print("\nRetrieved Context:\n", context)

# 7. Generate answer using Ollama
answer = requests.post(
    f"{OLLAMA_URL}/api/chat",
    json={
        "model": LLM_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "Answer only using the provided PDF context. If the answer is not in the context, say so."
            },
            {
                "role": "user",
                "content": f"PDF Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    }
)
answer.raise_for_status()

print("\nAI Answer:\n", answer.json()["message"]["content"])
