from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-exp",   # ✅ must include "models/"
    temperature=0.7,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
from langchain_core.messages import HumanMessage

result = chatbot.invoke(
    {"messages": [HumanMessage(content="Hello Gemini!")]},
    config={"configurable": {"thread_id": "thread-1"}}  # ✅ required
)

print(result["messages"][-1].content)
