from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class chat_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    

def chat_node(state:chat_state):
    messages=state["messages"]
    response=llm.invoke(messages)
    return {'messages':[response]}


graph=StateGraph(chat_state)
checkpointer=InMemorySaver()
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot=graph.compile(checkpointer=checkpointer)