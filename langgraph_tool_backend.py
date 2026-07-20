from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
from typing import Literal
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
    temperature=0,
)

# ==========================
# Tools
# ==========================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: Literal["add", "sub", "mul", "div"],
) -> dict:
    """
    Perform a basic arithmetic operation.
    Operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {"error": "Division by zero is not allowed"}

            result = first_num / second_num

        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price.
    """

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        "&apikey=C9PE94QUEW9VWGFM"
    )

    return requests.get(url).json()


tools = [
    search_tool,
    calculator,
    get_stock_price,
]

llm_with_tools = llm.bind_tools(tools)

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

    response = llm_with_tools.invoke(messages)

    print(response)

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)

# ==========================
# SQLite
# ==========================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)

# Store logical titles

conn.execute("""
CREATE TABLE IF NOT EXISTS thread_titles(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")

conn.commit()

checkpointer = SqliteSaver(conn)

# ==========================
# Graph
# ==========================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

chatbot = graph.compile(
    checkpointer=checkpointer
)

# =====================================================
# Thread Title Helpers
# =====================================================

def save_thread_title(thread_id: str, title: str):

    conn.execute(
        """
        INSERT OR REPLACE INTO thread_titles(thread_id,title)
        VALUES(?,?)
        """,
        (thread_id, title),
    )

    conn.commit()


def get_thread_title(thread_id: str):

    cursor = conn.execute(
        """
        SELECT title
        FROM thread_titles
        WHERE thread_id=?
        """,
        (thread_id,),
    )

    row = cursor.fetchone()

    if row:
        return row[0]

    return None


# =====================================================
# Retrieve Threads
# =====================================================

def retrieve_all_threads():

    threads = []

    seen = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id in seen:
            continue

        seen.add(thread_id)

        title = get_thread_title(thread_id)

        if title is None:
            title = "New Chat"

        threads.append(
            {
                "id": thread_id,
                "title": title,
            }
        )

    return threads