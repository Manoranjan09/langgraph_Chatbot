import streamlit as st
from langgrpah_backend import chatbot, llm
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************** Utility Functions ****************************

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(user_message):
    prompt = f"""
Generate a very short conversation title (maximum 4 words)
for the following user message.

Message:
{user_message}

Only return the title.
"""

    return llm.invoke(prompt).content.strip()


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id, "New Chat")


def add_thread(thread_id, title):

    for chat in st.session_state["chat_threads"]:
        if chat["id"] == thread_id:
            chat["title"] = title
            return

    st.session_state["chat_threads"].append(
        {
            "id": thread_id,
            "title": title,
        }
    )


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])


# **************************** Session Setup ****************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

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

        st.session_state["message_history"] = []

        for msg in messages:

            role = "user" if isinstance(msg, HumanMessage) else "assistant"

            st.session_state["message_history"].append(
                {
                    "role": role,
                    "content": msg.content,
                }
            )

        
# **************************** Chat UI ****************************

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here...")

if user_input:

    # Generate a logical title only once
    if len(st.session_state["message_history"]) == 0:

      title = generate_chat_title(user_input)

      add_thread(
        st.session_state["thread_id"],
        title
     )

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
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
            "content": ai_message
        }
    )