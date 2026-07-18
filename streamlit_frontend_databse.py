import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads, llm
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************** Utility Functions ****************************

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(user_message):
    prompt = f"""
Generate a short chat title (maximum 4 words) for the following message.

Message:
{user_message}

Only return the title.
"""
    return llm.invoke(prompt).content.strip()


def add_thread(thread_id, title=None):

    for chat in st.session_state["chat_threads"]:

        if chat["id"] == thread_id:

            if title is not None:
                chat["title"] = title

            return

    st.session_state["chat_threads"].append(
        {
            "id": thread_id,
            "title": title or "New Chat"
        }
    )


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# **************************** Session Setup ****************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = []

    # Load old conversations from SQLite
    for thread in retrieve_all_threads():
        st.session_state["chat_threads"].append(
            {
                "id": thread,
                "title": "Previous Chat"
            }
        )

current_thread_exists = any(
    chat["id"] == st.session_state["thread_id"]
    for chat in st.session_state["chat_threads"]
)

if not current_thread_exists:
    add_thread(st.session_state["thread_id"], "New Chat")


# **************************** Sidebar ****************************

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for chat in reversed(st.session_state["chat_threads"]):

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

            temp_messages.append(
                {
                    "role": role,
                    "content": msg.content,
                }
            )

        st.session_state["message_history"] = temp_messages


# **************************** Main UI ****************************

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here...")

if user_input:

    # Generate title only once
    if len(st.session_state["message_history"]) == 0:

        title = generate_chat_title(user_input)

        add_thread(
            st.session_state["thread_id"],
            title
        )

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        def stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(stream())

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )