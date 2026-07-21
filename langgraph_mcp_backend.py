from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
import os
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Literal
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

import aiosqlite
import asyncio
import threading
import requests

load_dotenv()

# =====================================================
# Dedicated Async Event Loop
# =====================================================

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True,
)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# =====================================================
# Tools
# =====================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


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


client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args":  ["/Users/manoranjankumar/Documents/mcp-math-server/# main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
    }
)


def load_mcp_tools() -> list[BaseTool]:

    try:
        return run_async(client.get_tools())

    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [
    search_tool,
    get_stock_price,
    *mcp_tools,
]

llm_with_tools = llm.bind_tools(tools) if tools else llm

# =====================================================
# State
# =====================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =====================================================
# Chat Node
# =====================================================

async def chat_node(state: ChatState):

    response = await llm_with_tools.ainvoke(state["messages"])

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools) if tools else None

# =====================================================
# Database + Checkpointer
# =====================================================

db_conn = None


async def _init_checkpointer():

    global db_conn

    db_conn = await aiosqlite.connect("chatbot.db")

    await db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_titles(
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
        """
    )

    await db_conn.commit()

    return AsyncSqliteSaver(db_conn)


checkpointer = run_async(_init_checkpointer())

# =====================================================
# Graph
# =====================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(
    START,
    "chat_node",
)

if tool_node:

    graph.add_node(
        "tools",
        tool_node,
    )

    graph.add_conditional_edges(
        "chat_node",
        tools_condition,
    )

    graph.add_edge(
        "tools",
        "chat_node",
    )

else:

    graph.add_edge(
        "chat_node",
        END,
    )


chatbot = graph.compile(
    checkpointer=checkpointer
)

# =====================================================
# Thread Title Helpers
# =====================================================

async def _save_thread_title(
    thread_id: str,
    title: str,
):

    await db_conn.execute(
        """
        INSERT OR REPLACE INTO thread_titles(thread_id,title)
        VALUES(?,?)
        """,
        (
            thread_id,
            title,
        ),
    )

    await db_conn.commit()


def save_thread_title(
    thread_id: str,
    title: str,
):

    run_async(
        _save_thread_title(
            thread_id,
            title,
        )
    )


async def _get_thread_title(
    thread_id: str,
):

    cursor = await db_conn.execute(
        """
        SELECT title
        FROM thread_titles
        WHERE thread_id=?
        """,
        (thread_id,),
    )

    row = await cursor.fetchone()

    if row:
        return row[0]

    return None


def get_thread_title(
    thread_id: str,
):

    return run_async(
        _get_thread_title(thread_id)
    )


# =====================================================
# Retrieve Threads
# =====================================================

async def _alist_threads():

    threads = []

    seen = set()

    async for checkpoint in checkpointer.alist(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id in seen:
            continue

        seen.add(thread_id)

        title = await _get_thread_title(thread_id)

        if title is None:
            title = "New Chat"

        threads.append(
            {
                "id": thread_id,
                "title": title,
            }
        )

    return threads


def retrieve_all_threads():

    return run_async(
        _alist_threads()
    )