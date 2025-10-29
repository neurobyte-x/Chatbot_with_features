import streamlit as st
from langgraph_backend import chatbot, generate_chat_name_from_message, save_uploaded_image
from langchain_core.messages import HumanMessage, AIMessage
import uuid

st.set_page_config(page_title="LangGraph AI Agent", layout="wide", page_icon="🤖")

# Title with file upload at the right corner
title_col1, title_col2 = st.columns([0.85, 0.15])
with title_col1:
    st.title("🤖 LangGraph AI Agent")
    st.caption("An intelligent agent with 14 powerful tools for web search, code execution, file management, and more!")
with title_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    uploaded_file = st.file_uploader(
        "📎 Attach Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for analysis"
    )
    
# Handle image upload
if uploaded_file is not None:
    filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = save_uploaded_image(uploaded_file, filename)
    st.session_state['uploaded_image_path'] = file_path
    st.success(f"✅ {uploaded_file.name}")
elif st.session_state.get('uploaded_image_path'):
    # Show currently attached image
    image_name = st.session_state['uploaded_image_path'].split('\\')[-1]
    st.info(f"🖼️ {image_name}")

# --- Helper Functions ---
def generate_thread():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread()
    st.session_state['thread_id'] = thread_id
    st.session_state['msg_history'] = []
    st.session_state['chat_names'][thread_id] = f"Chat {thread_id[:8]}"
    st.session_state['user_messages'][thread_id] = []
    st.session_state['uploaded_image_path'] = None  # Clear uploaded image
    add_thread(thread_id)
    return thread_id

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    temp_msg = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        temp_msg.append({'role': role, 'content': msg.content})
    return temp_msg

# --- Initialize Session State ---
for key, default in {
    'msg_history': [],
    'chat_names': {},
    'thread_id': generate_thread(),
    'chat_threads': [],
    'user_messages': {},
    'uploaded_image_path': None,
    'last_input': ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

add_thread(st.session_state['thread_id'])

# --- Sidebar ---
st.sidebar.title("🤖 AI Agent")
st.sidebar.caption("v1.0 - 14 Tools Available")
if st.sidebar.button('New Chat'):
    reset_chat()

# Available Tools Info
st.sidebar.header("🛠️ Available Tools")
with st.sidebar.expander("View All Tools"):
    st.markdown("""
    **Core Tools:**
    - 🔍 Web Search
    - 🐍 Python Executor
    - 🧮 Calculator
    - 🌤️ Weather Info
    - 🖼️ Image Analysis
    
    **Browser Automation:**
    - 🌐 Navigate Webpages 👈 new
    - 🖱️ Click Elements 👈 new
    - 🔗 Extract Links 👈 new
    - 📸 Screenshot Pages 👈 new
    
    **Web & Data:**
    - 🌐 Search StackOverflow
    - 💾 SQL Queries
    
    **Utilities:**
    - 🔢 Unit Conversion
    """)

st.sidebar.header("My Conversations")
for idx, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    display_name = st.session_state['chat_names'].get(thread_id, str(thread_id))
    if st.sidebar.button(display_name, key=f"{thread_id}_{idx}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['msg_history'] = load_conversation(thread_id)

# --- Display Messages ---
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input
user_input = st.chat_input("💬 Type your message here...")

if user_input:
    thread_id = st.session_state['thread_id']

    # Track user messages
    if thread_id not in st.session_state['user_messages']:
        st.session_state['user_messages'][thread_id] = []
    st.session_state['user_messages'][thread_id].append(user_input)

    # Check if there's an uploaded image
    image_context = ""
    if st.session_state['uploaded_image_path']:
        image_context = f"\n[Image uploaded at: {st.session_state['uploaded_image_path']}]"
        # Always include image analysis instruction when an image is uploaded
        user_input_with_image = f"{user_input}\n\n[SYSTEM: User has uploaded an image. Use the image_reasoning_tool to analyze the image at: {st.session_state['uploaded_image_path']} and incorporate the analysis into your response.]"
    else:
        user_input_with_image = user_input

    # Append user message to history (display version without image path)
    st.session_state['msg_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        if image_context:
            st.caption("🖼️ Image attached")

    # Prepare config for chatbot
    config = {'configurable': {'thread_id': thread_id}}

    # Stream AI response (send version with image path to chatbot)
    with st.chat_message('assistant'):
        ai_content = ""
        for msg_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input_with_image)]},
            config=config,
            stream_mode="messages"
        ):
            if isinstance(msg_chunk, AIMessage):
                # Handle both string and list content
                chunk_content = msg_chunk.content
                if isinstance(chunk_content, list):
                    # Extract text from list format (sometimes happens with tool calls)
                    chunk_content = " ".join(str(item) for item in chunk_content if item)
                elif chunk_content is None:
                    chunk_content = ""
                
                ai_content += chunk_content
                if chunk_content:  # Only write non-empty content
                    st.write(chunk_content)
        
        # Check if response contains a screenshot path and display it
        if "uploaded_images" in ai_content and "screenshot_" in ai_content and ".png" in ai_content:
            import re
            screenshot_match = re.search(r'uploaded_images[/\\]screenshot_\d+\.png', ai_content)
            if screenshot_match:
                screenshot_path = screenshot_match.group(0)
                try:
                    st.image(screenshot_path, caption="Screenshot", use_container_width=True)
                except Exception:
                    pass

    # Append AI message to history
    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_content})

    # Generate chat name if first two messages exist
    if len(st.session_state['user_messages'][thread_id]) >= 2:
        first_two = " ".join(st.session_state['user_messages'][thread_id][:2])
        name = generate_chat_name_from_message(first_two)
        st.session_state['chat_names'][thread_id] = f"{name}_{thread_id[:8]}"
