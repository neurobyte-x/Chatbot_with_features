from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langsmith import traceable


load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class chat_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
@traceable(name="chat_node")
def chat_node(state:chat_state):
    messages=state["messages"]
    response=llm.invoke(messages)
    return {'messages':[response]}

from langchain_core.messages import HumanMessage

def generate_chat_name_from_message(first_message):
    prompt = f"Generate a short 3–5 word chat title summarizing this message:\n\"{first_message}\""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def generate_chat_name_safe(thread_id, first_message):
    name = generate_chat_name_from_message(first_message)
    return f"{name}_{str(thread_id)[:8]}"



graph=StateGraph(chat_state)
checkpointer=InMemorySaver()
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot=graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)