
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# 1. Initialize Ollama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)

# 2. Create Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question"
)

# 3. Create Prompt Template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a helpful AI assistant.
Use the previous conversation to answer questions.

Conversation History:
{chat_history}

User: {question}
Assistant:
"""
)

# 4. Create conversation function
def chat(question):
    # Load previous conversation
    history = memory.load_memory_variables({})

    # Prepare prompt
    formatted_prompt = prompt.format(
        chat_history=history["chat_history"],
        question=question
    )

    # Generate response
    response = llm.invoke(formatted_prompt)

    answer = response.content

    # Save current conversation
    memory.save_context(
        {"question": question},
        {"output": answer}
    )

    return answer


# 5. Test five conversation turns
questions = [
    "My name is Ameesha.",
    "I am learning Python.",
    "What is my name?",
    "What am I learning?",
    "Introduce me using the information I shared."
]

for i, question in enumerate(questions, start=1):
    print(f"\nTurn {i}: {question}")
    answer = chat(question)
    print("AI:", answer)

# 6. Verify memory
print("\nTotal conversation turns:", len(questions))
print("\nStored Conversation History:")
print(memory.load_memory_variables({})["chat_history"])

print("\nConversationBufferMemory test completed!")

