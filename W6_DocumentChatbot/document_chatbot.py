from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb

PDF_PATH = "W6_ChromaDB/sample.pdf"

# Load PDF
reader = PdfReader(PDF_PATH)
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# Split into chunks
chunk_size = 500
overlap = 100
chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size - overlap)
    if text[i:i + chunk_size].strip()
]

# Create embeddings
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
vectors = embeddings_model.embed_documents(chunks)

# Store in ChromaDB
client = chromadb.PersistentClient(path="./W6_DocumentChatbot/chroma_db")
collection = client.get_or_create_collection(
    name="document_chatbot",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=vectors
)

# Load LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

print("Document Chatbot Ready! Ask questions about your PDF.")
print("Type 'exit' to stop.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Chatbot closed.")
        break

    if not question:
        continue

    query_vector = embeddings_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(3, collection.count())
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
    Answer using only the document context below.
    If the answer is unavailable, say it is not found in the document.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    response = llm.invoke(prompt)
    print("\nChatbot:", response.content)
