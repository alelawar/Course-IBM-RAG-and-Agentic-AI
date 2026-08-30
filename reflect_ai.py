import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, MessageGraph
from typing import List, Sequence

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post's overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post's potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post's relevance to the author's industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post's strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflect_chain = reflection_prompt | llm

graph = MessageGraph()

def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    generated_post = generate_chain.invoke({"messages": state})
    return [AIMessage(content=generated_post.content)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Tukar role: AIMessage jadi HumanMessage dan sebaliknya,
    # supaya giliran terakhir yang dikirim ke Gemini selalu dari "user"
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = reflect_chain.invoke({"messages": translated})
    return [HumanMessage(content=res.content)]

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return "reflect"

graph.add_conditional_edges("generate", should_continue)

workflow = graph.compile()

inputs = HumanMessage(content="Write a linkedin post on getting a software developer job at IBM under 160 characters")

response = workflow.invoke(inputs)

print("--- Draft pertama (sebelum kritik) ---")
print(response[1].content)

print("\n--- Kritik pertama ---")
print(response[2].content)

print("\n--- Hasil akhir setelah beberapa iterasi ---")
print(response[-1].content)

