# W6D5 - Document Chatbot using LangChain

## Objective
Build a PDF-based chatbot using LangChain, ChromaDB and Ollama.

## Tasks Completed
1. Loaded PDF using pypdf.
2. Extracted PDF text.
3. Split text into chunks.
4. Generated embeddings using nomic-embed-text.
5. Stored document chunks in ChromaDB.
6. Retrieved relevant chunks for user questions.
7. Passed retrieved context to llama3.2:3b.
8. Generated answers using document context.
9. Tested chatbot with two questions.

## Technologies
- Python 3.12
- LangChain Ollama
- ChromaDB
- Ollama
- pypdf

## How to Run
python W6_DocumentChatbot/document_chatbot.py

## Evidence
screenshots/W6D5/chatbot_output.png

## Result
Document chatbot successfully answered questions using PDF content.
