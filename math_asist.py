from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import re, os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# response = llm.invoke("Apa itu tool calling di LangChain?")
# # print(response.text)

@tool
def add_numbers(inputs: str) -> dict:
    """Adds all numbers found in the input string and returns their sum."""
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    return {"result": sum(numbers)}

@tool
def new_subtract_numbers(inputs: str) -> dict:
    """Extracts numbers from the input string and subtracts them sequentially, starting from the first number."""
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}

@tool
def multiply_numbers(inputs: str) -> dict:
    """Extracts numbers from the input string and calculates their product."""
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 1}
    result = 1
    for num in numbers:
        result *= num
    return {"result": result}

@tool
def divide_numbers(inputs: str) -> dict:
    """Extracts numbers from the input string and divides the first number by each subsequent number in sequence."""
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result /= num
    return {"result": result}

tools = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]

math_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)

response = math_agent.invoke({
    "messages": [("human", "What is 25 divided by 4?")]
})
print(response["messages"][-1].content)