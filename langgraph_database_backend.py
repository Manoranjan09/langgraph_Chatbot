from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq

from dotenv import load_dotenv
import sqlite3
import os

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ==========================
# Groq LLM
# ==========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ==========================
# State
# ==========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================
# Chat Node
# ==========================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }

# ==========================
# SQLite Checkpointer
# ==========================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

# ==========================
# Graph
# ==========================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ==========================
# Compile Graph
# ==========================

chatbot = graph.compile(
    checkpointer=checkpointer
)

# ==========================
# Retrieve All Threads
# ==========================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)