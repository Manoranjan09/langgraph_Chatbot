import queue
import uuid

import streamlit as st
from langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    save_thread_title,
    llm,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
def generate_chat_title(user_message):

    prompt = f"""
Generate a very short conversation title
(maximum 4 words).

Message:
{user_message}

Only return the title.
"""

    return llm.invoke(prompt).content.strip()
# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id, "New Chat")
    st.session_state["message_history"] = []


def add_thread(thread_id, title="New Chat"):

    for i, chat in enumerate(st.session_state["chat_threads"]):

        if chat["id"] == thread_id:

            chat["title"] = title

            st.session_state["chat_threads"].pop(i)
            st.session_state["chat_threads"].insert(0, chat)

            return

    st.session_state["chat_threads"].insert(
        0,
        {
            "id": thread_id,
            "title": title,
        },
    )


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

current_exists = any(
    chat["id"] == st.session_state["thread_id"]
    for chat in st.session_state["chat_threads"]
)

if not current_exists:
    add_thread(
        st.session_state["thread_id"],
        "New Chat",
    )

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for chat in st.session_state["chat_threads"]:

    if st.sidebar.button(
        chat["title"],
        key=f"chat_{chat['id']}",
        use_container_width=True,
    ):

        st.session_state["thread_id"] = chat["id"]

        messages = load_conversation(chat["id"])

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    if len(st.session_state["message_history"]) == 0:

    title = generate_chat_title(user_input)

    save_thread_title(
        st.session_state["thread_id"],
        title,
    )

    add_thread(
        st.session_state["thread_id"],
        title,
    )
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )