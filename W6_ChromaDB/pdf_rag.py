from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
import chromadb

# Read PDF
reader = PdfReader("W6_ChromaDB/sample.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# Split text into chunks
chunk_size = 500
overlap = 100
chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size - overlap)
    if text[i:i + chunk_size].strip()
]

print("PDF pages:", len(reader.pages))
print("Text chunks:", len(chunks))

# Create embeddings
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
embeddings = embedding_model.embed_documents(chunks)

# Store chunks in a separate W6 collection
client = chromadb.PersistentClient(path="./W6_ChromaDB/chroma_db")
collection = client.get_or_create_collection(
    name="pdf_rag_docs",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[f"pdf_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": "sample.pdf", "chunk": i} for i in range(len(chunks))]
)

print("PDF chunks stored:", collection.count())
print("PDF ingestion completed successfully!")
