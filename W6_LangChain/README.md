# Week 6 Day 1 - LangChain Fundamentals

## Objective
To understand and implement LangChain chains, conversation memory, and agents using local Ollama models.

## 1. Chain Implementation
File: chain_testing.py

- Used PromptTemplate to create dynamic prompts.
- Used Ollama LLM with llama3.2:3b.
- Used StrOutputParser to extract text output.
- Tested the chain with 5 different topics.

## 2. Conversation Memory
File: memory.py

- Implemented ConversationBufferMemory.
- Tested memory across 5 conversation turns.
- Verified that the model could recall the user's name and learning topic.

## 3. Agent Implementation
File: agent.py

Created an agent with two tools:

1. Calculator - performs mathematical calculations.
2. Web Search Stub - provides sample search information.

Tested the agent with 3 tasks successfully.

## Technologies Used

- Python
- LangChain
- LangChain Classic
- Ollama
- Llama 3.2
- Qwen 2.5

## How to Run

```bash
python chain_testing.py
python memory.py
python agent.py