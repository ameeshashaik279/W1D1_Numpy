from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

client = chromadb.PersistentClient(path="./W6_ChromaDB/chroma_db")
collection = client.get_collection("pdf_rag_docs")

question = "What is Retrieval Augmented Generation (RAG)?"

query_embedding = embedding_model.embed_query(question)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

context = "\n".join(results["documents"][0])

prompt = f"""
Answer the question using only the context below.
If the answer is not present, say you cannot find it in the PDF.

Context:
{context}

Question: {question}
Answer:
"""

response = llm.invoke(prompt)

print("\nQUESTION:", question)
print("\nRETRIEVED CONTEXT:\n", context)
print("\nLLM ANSWER:\n", response.content)
