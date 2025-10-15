import streamlit as st
from langgraph_backend import chatbot, generate_chat_name_from_message
from langchain_core.messages import HumanMessage, AIMessage
import uuid

st.title("LangGraph Chatbot")


def generate_thread():
    return uuid.uuid4()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread()
    st.session_state['thread_id'] = thread_id
    st.session_state['msg_history'] = []
    st.session_state['chat_names'][thread_id] = f"Chat {str(thread_id)[:8]}"  # temporary name
    st.session_state['user_messages'][thread_id] = []  # keep track of messages for naming
    add_thread(thread_id)
    return thread_id

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# --- Initialize session state ---
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []

if 'chat_names' not in st.session_state:
    st.session_state['chat_names'] = {}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'user_messages' not in st.session_state:
    st.session_state['user_messages'] = {}

add_thread(st.session_state['thread_id'])


st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("My Conversations")
for idx, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    display_name = st.session_state['chat_names'].get(thread_id, str(thread_id))
    if st.sidebar.button(display_name, key=f"{thread_id}_{idx}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_msg = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_msg.append({'role': role, 'content': msg.content})
        st.session_state['msg_history'] = temp_msg


for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])


user_input = st.chat_input("Type Your Query Here:")
if user_input:
    thread_id = st.session_state['thread_id']

 
    if thread_id not in st.session_state['user_messages']:
        st.session_state['user_messages'][thread_id] = []
    st.session_state['user_messages'][thread_id].append(user_input)


    st.session_state['msg_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    config = {'configurable': {'thread_id': thread_id}}

    with st.chat_message('assistant'):
        def ai_only_stream():
            for msg_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(msg_chunk, AIMessage):
                    yield msg_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_message})

    if len(st.session_state['user_messages'][thread_id]) >= 2:
        first_two = " ".join(st.session_state['user_messages'][thread_id][:2])
        name = generate_chat_name_from_message(first_two)
        st.session_state['chat_names'][thread_id] = f"{name}_{str(thread_id)[:8]}"