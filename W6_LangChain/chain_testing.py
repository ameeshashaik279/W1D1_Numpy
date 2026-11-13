from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Create Prompt Template
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words with one example."
)

# 2. Connect to Ollama LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)

# 3. Create Output Parser
parser = StrOutputParser()

# 4. Build LangChain
chain = prompt | llm | parser

# 5. Test with 5 inputs
topics = [
    "Machine Learning",
    "Artificial Intelligence",
    "Python",
    "Data Science",
    "Cloud Computing"
]

for topic in topics:
    print(f"\nTopic: {topic}")
    response = chain.invoke({"topic": topic})
    print("Response:", response)
    print("-" * 50)