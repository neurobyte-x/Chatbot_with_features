import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage,AIMessage
import uuid

st.title("LangGraph Chatbot")

def generate_thread():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=generate_thread()
    st.session_state['thread_id']=thread_id
    st.session_state['msg_history']=[]
    
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state=chatbot.get_state(config={'configurable': {'thread_id':thread_id}})
    return state.values.get('messages',[])



if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
    
add_thread(st.session_state['thread_id'])
    
st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button('New Chat'):
    reset_chat()
    
    
    
st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id)
        
        temp_msg=[]
        
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_msg.append({'role':role,'content':msg.content})
        st.session_state['msg_history']=temp_msg

for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])


user_input=st.chat_input("Type Your Query Here:")

if user_input:
    
    st.session_state['msg_history'].append({'role':'user','content':user_input})
    
    with st.chat_message('user'):
        st.text(user_input)
        
    config1={'configurable': {'thread_id': st.session_state['thread_id']}}
    
    response = chatbot.invoke(
        {'messages':[HumanMessage(content=user_input)]},
        config=config1
    )
    # ai_message=response['messages'][-1].content
    # st.session_state['msg_history'].append({'role':'assistant','content':ai_message})
    with st.chat_message('assistant'):
        def ai_only_stream():
            for msg_chunk,metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=config1,
                stream_mode="messages"
            ):
                if isinstance(msg_chunk,AIMessage):
                    yield msg_chunk.content
        ai_message = st.write_stream(ai_only_stream())

    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_message})
