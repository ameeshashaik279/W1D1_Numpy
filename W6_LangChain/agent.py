
import ast
import operator

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# ==========================================
# 1. SAFE CALCULATOR
# ==========================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def calculate_expression(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](
                evaluate(node.left),
                evaluate(node.right)
            )

        raise ValueError("Unsupported mathematical expression")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


# ==========================================
# 2. TOOL 1 - CALCULATOR
# ==========================================

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions like 25 * 4 or 100 / 5."""

    try:
        result = calculate_expression(expression)
        return str(result)

    except Exception as error:
        return f"Calculation error: {error}"


# ==========================================
# 3. TOOL 2 - WEB SEARCH STUB
# ==========================================

@tool
def web_search(query: str) -> str:
    """Search sample information about Python, Machine Learning and Cloud Computing. This is not live web search."""

    sample_data = {
        "python": "Python is a popular programming language used for AI, automation and web development.",

        "machine learning": "Machine Learning allows computers to learn patterns from data.",

        "cloud computing": "Cloud computing provides computing services over the internet."
    }

    query_lower = query.lower()

    for keyword, result in sample_data.items():

        if keyword in query_lower:
            return result

    return "No matching sample information found."


# ==========================================
# 4. INITIALIZE OLLAMA LLM
# ==========================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ==========================================
# 5. CREATE LANGCHAIN AGENT
# ==========================================

tools = [
    calculator,
    web_search
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the calculator tool for mathematical calculations. "
        "Use the web_search tool for questions about the sample knowledge base. "
        "Give clear and accurate answers."
    )
)


# ==========================================
# 6. TEST THREE AGENT TASKS
# ==========================================

tasks = [
    "Calculate 25 multiplied by 16.",

    "Search for information about Python.",

    "Calculate 150 divided by 5 and tell me about cloud computing."
]


# ==========================================
# 7. EXECUTE AND DISPLAY RESULTS
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("LANGCHAIN TWO-TOOL AGENT")
    print("=" * 55)

    print("\nAvailable Tools:")
    for item in tools:
        print("-", item.name)

    for i, task in enumerate(tasks, start=1):

        print(f"\nTask {i}: {task}")

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": task}
            ]
        })

        final_message = result["messages"][-1]

        print("\nAgent Response:")
        print(final_message.content)

        print("-" * 55)

    print("\nAll 3 Agent Tasks Completed!")
