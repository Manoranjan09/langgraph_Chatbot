import streamlit as st
from langgrpah_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

# Initialize chat history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type here...")

if user_input:

    # Display and store user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):

        def stream_response():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(stream_response())

    # Store assistant response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )