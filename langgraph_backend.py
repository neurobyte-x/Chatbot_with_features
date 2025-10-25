from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import re
import math
import requests
import os
from langchain_community.tools import DuckDuckGoSearchRun
try:
    from langchain_experimental.utilities import PythonREPL
except ImportError:
    from langchain_community.utilities import PythonREPL
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import shutil
from bs4 import BeautifulSoup
import sqlite3
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)



class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



@tool
def web_search(query: str) -> str:
    """Use this tool to search the web for relevant information."""
    try:
        search = DuckDuckGoSearchRun()
        res = search.run(query)
        return f'Web search results for "{query}":\n{res}'
    except Exception as e:
        return f'Error occurred: {str(e)}'


@tool
def python_code_executor(code: str) -> str:
    """Execute Python code and return the result."""
    try:
        try:
            executor = PythonREPL()
            result = executor.run(code)
        except Exception:
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code)
            result = output.getvalue()
        
        return f'Code execution result:\n{result}'
    except Exception as e:
        return f'Error executing code: {str(e)}'


@tool
def mathematical_calculator(expression: str) -> str:
    """Perform mathematical calculations (arithmetic, trig, log, etc.)."""
    try:
        expression = re.sub(r'[^0-9+\-*/().\s,]', '', expression)
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation failed: {str(e)}"


@tool
def weather_tool(city: str) -> str:
    """Get current weather information for a specified city."""
    try:
        weather = OpenWeatherMapAPIWrapper()
        weather_info = weather.run(city)
        return f'Weather information for {city}:\n{weather_info}'
    except Exception as e:
        return f'Error occurred: {str(e)}'




@tool
def image_reasoning_tool(image_path: str) -> str:
    """
    Uses Gemini 2.5 Pro to analyze an image and detect broken or damaged objects.
    Especially checks for issues in fans, lights, furniture, or electronics.
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel("gemini-2.5-pro")

        image = Image.open(image_path)

        prompt = (
            "You are a visual inspector. Analyze this image carefully and describe any broken, "
            "damaged, or malfunctioning items you observe, focusing on electrical items "
            "like fans, lights, or appliances. Be specific and objective."
        )

        response = model.generate_content([prompt, image])

        return response.text.strip() if response.text else "No visible issues detected."
    
    except Exception as e:
        return f"Error occurred: {str(e)}"



@tool
def create_file(file_path: str, content: str) -> str:
    """Create a new file with the given content."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ File created: {file_path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def read_file_tool(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content ({file_path}):\n{content[:2000]}..."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """List all files and folders in a directory."""
    try:
        path = Path(directory_path)
        items = list(path.iterdir())
        
        result = f"📁 Contents of {directory_path}:\n"
        for item in items:
            icon = "📁" if item.is_dir() else "📄"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            result += f"{icon} {item.name}{size}\n"
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file."""
    try:
        shutil.move(source, destination)
        return f"✅ Moved {source} to {destination}"
    except Exception as e:
        return f"Error moving file: {str(e)}"


@tool
def delete_file(file_path: str) -> str:
    """Delete a file."""
    try:
        os.remove(file_path)
        return f"✅ Deleted {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


# ---------- WEB SCRAPING TOOLS ----------
@tool
def scrape_webpage(url: str) -> str:
    """Extract text content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return f"Content from {url}:\n{text[:2000]}..."
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"


@tool
def extract_links(url: str) -> str:
    """Extract all links from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        for link in soup.find_all('a', href=True):
            text = link.text.strip() or "[No text]"
            links.append(f"{text}: {link['href']}")
        
        return f"Links found on {url}:\n" + "\n".join(links[:30])
    except Exception as e:
        return f"Error extracting links: {str(e)}"



@tool
def execute_sql_query(database_path: str, query: str) -> str:
    """Execute a SQL query on a SQLite database."""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(query)
        
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            
            formatted_results = []
            for row in results[:50]:
                formatted_results.append(dict(zip(columns, row)))
            
            return f"Query results:\n{json.dumps(formatted_results, indent=2)}"
        else:
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            return f"✅ Query executed successfully: {rows_affected} rows affected"
    except Exception as e:
        return f"Error executing query: {str(e)}"



@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units (temperature, length, weight, etc.)."""
    try:
        conversions = {
            ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
            ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
            ('celsius', 'kelvin'): lambda x: x + 273.15,
            ('kelvin', 'celsius'): lambda x: x - 273.15,
            ('meters', 'feet'): lambda x: x * 3.28084,
            ('feet', 'meters'): lambda x: x / 3.28084,
            ('kilometers', 'miles'): lambda x: x * 0.621371,
            ('miles', 'kilometers'): lambda x: x / 0.621371,
            ('centimeters', 'inches'): lambda x: x * 0.393701,
            ('inches', 'centimeters'): lambda x: x / 0.393701,
            ('kilograms', 'pounds'): lambda x: x * 2.20462,
            ('pounds', 'kilograms'): lambda x: x / 2.20462,
            ('grams', 'ounces'): lambda x: x * 0.035274,
            ('ounces', 'grams'): lambda x: x / 0.035274,
        }
        
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        else:
            return f"Conversion from {from_unit} to {to_unit} not supported. Available: celsius, fahrenheit, kelvin, meters, feet, kilometers, miles, centimeters, inches, kilograms, pounds, grams, ounces"
    except Exception as e:
        return f"Error converting units: {str(e)}"


tools = [
    web_search,
    python_code_executor,
    mathematical_calculator,
    weather_tool,
    image_reasoning_tool,
    create_file,
    read_file_tool,
    list_directory,
    move_file,
    delete_file,
    scrape_webpage,
    extract_links,
    execute_sql_query,
    convert_units
]



@traceable(name="chat_node")
def chat_node(state: chat_state):
    messages = state["messages"]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}



def generate_chat_name_from_message(first_message):
    prompt = f"Generate a short 3–5 word chat title summarizing this message:\n\"{first_message}\""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def generate_chat_name_safe(thread_id, first_message):
    name = generate_chat_name_from_message(first_message)
    return f"{name}_{str(thread_id)[:8]}"



graph = StateGraph(chat_state)
checkpointer = InMemorySaver()


tool_node = ToolNode(tools)


graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)


graph.add_edge(START, 'chat_node')

graph.add_conditional_edges('chat_node', tools_condition)

graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpointer)



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)



def save_uploaded_image(uploaded_file, filename):
    """Save uploaded image file and return the file path."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path
