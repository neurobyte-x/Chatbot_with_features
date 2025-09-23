import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

config1={'configurable': {'thread_id': 'thread-1'}}

if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]


for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])


user_input=st.chat_input("Type Your Query Here:")

if user_input:
    
    st.session_state['msg_history'].append({'role':'user','content':user_input})
    
    with st.chat_message('user'):
        st.text(user_input)
    
    response = chatbot.invoke(
        {'messages':[HumanMessage(content=user_input)]},
        config=config1
    )
    # ai_message=response['messages'][-1].content
    # st.session_state['msg_history'].append({'role':'assistant','content':ai_message})
    with st.chat_message('assistant'):
        ai_msg=st.write_stream(
            msg_chunk.content for msg_chunk,metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=config1,
                stream_mode='messages'
            )
        )
    st.session_state['msg_history'].append({'role':'assistant','content':ai_msg})