
import requests

OLLAMA_API = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer clearly in simple English.
Use examples wherever useful.
"""

prompts = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning with an example.",
    "What is Python used for?",
    "Explain SQL joins in simple English.",
    "What is the difference between AI and ML?"
]

comparison_questions = [
    "What is Data Science?",
    "Explain supervised learning with an example.",
    "Why is data cleaning important?"
]

def ask_model(model, prompt):
    response = requests.post(
        OLLAMA_API,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

with open("ollama_results.txt", "w", encoding="utf-8") as file:

    file.write("FIVE PROMPT TESTS - LLAMA 3.2\n\n")

    for i, prompt in enumerate(prompts, 1):
        answer = ask_model("llama3.2:3b", prompt)
        file.write(f"Prompt {i}: {prompt}\n")
        file.write(f"Answer: {answer}\n\n")
        print(f"Llama prompt {i}/5 completed")

    file.write("\nMODEL COMPARISON\n")

    for i, question in enumerate(comparison_questions, 1):
        llama_answer = ask_model("llama3.2:3b", question)
        qwen_answer = ask_model("qwen2.5:3b", question)

        file.write(f"\nQuestion {i}: {question}\n")
        file.write(f"\nLlama 3.2:\n{llama_answer}\n")
        file.write(f"\nQwen 2.5:\n{qwen_answer}\n")

        print(f"Comparison {i}/3 completed")

print("\nAll tests completed. Check ollama_results.txt")
