import math
import os
import re
from typing import Annotated, TypedDict
import asyncio
import json
import sqlite3
import threading
import nest_asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from langchain_community.utilities import (
    GoogleSerperAPIWrapper,
    OpenWeatherMapAPIWrapper,
)
from PIL import Image

try:
    from playwright.sync_api import sync_playwright
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from langchain_experimental.utilities import PythonREPL
except ImportError:
    from langchain_community.utilities import PythonREPL


nest_asyncio.apply()

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

try:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
except AttributeError:
    pass

UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)



class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



@tool
def web_search(query: str) -> str:
    """Search the web using Google Serper API and return summarized results."""
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: SERPER_API_KEY not found in environment variables."

        search = GoogleSerperAPIWrapper(serper_api_key=api_key)
        results = search.run(query)

        return f"🔍 Web search results for '{query}':\n\n{results.strip()}"
    except Exception as e:
        return f"❌ Error in web_search tool: {str(e)}"


@tool
def python_code_executor(code: str) -> str:
    """Execute Python code and return the result."""
    try:
        try:
            executor = PythonREPL()
            result = executor.run(code)
        except Exception:
            import contextlib
            import io
            
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
    Analyze any image using Gemini 2.5 Pro vision capabilities.
    Can analyze screenshots, photos, diagrams, code snippets, UI elements, 
    LeetCode problems, documents, broken items, and more.
    Provides detailed description of the content.
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel("gemini-2.5-pro")

        image = Image.open(image_path)

        prompt = (
            "Analyze this image in detail. Describe what you see, including:\n"
            "- Main content and purpose\n"
            "- Any text, code, or problems visible\n"
            "- UI elements, screenshots, or interfaces\n"
            "- Technical details if it's code/programming related\n"
            "- Any issues, errors, or problems to solve\n"
            "Be thorough, specific, and helpful in your analysis."
        )

        response = model.generate_content([prompt, image])

        return response.text.strip() if response.text else "Unable to analyze the image."
    
    except Exception as e:
        return f"Error occurred: {str(e)}"
    
    




def run_playwright_navigation(url: str, selector: str = "body") -> str:
    """Worker function to isolate Playwright in its own thread."""
    if not PLAYWRIGHT_AVAILABLE:
        return "Error: Playwright is not installed. Browser automation is unavailable."
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.wait_for_selector(selector, timeout=10000)
            elements = page.query_selector_all(selector)
            texts = [el.inner_text() for el in elements]
            browser.close()
            return "\n\n".join(texts[:10])
    except Exception as e:
        return f"Playwright error: {str(e)}"

@tool
def browser_navigation(url: str, selector: str = "body") -> str:
    """
    Navigate to a URL and extract text from a selector.
    Note: This tool may not work in cloud deployment environments without proper browser setup.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "❌ Browser automation not available. Playwright is not installed."
    
    result_holder = {}

    def run():
        result_holder["res"] = run_playwright_navigation(url, selector)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return result_holder.get("res", "No result.")



@tool
def click_element(selector: str, url: str) -> str:
    """
    Click an element on a web page using a CSS selector.
    Note: This tool may not work in cloud deployment environments without proper browser setup.

    Args:
        selector (str): The CSS selector of the element to click.
        url (str): The webpage URL to visit before clicking.

    Returns:
        str: Status message indicating the click result.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "❌ Browser automation not available. Playwright is not installed."
    
    async def _click():
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30000)
                await page.wait_for_selector(selector, timeout=10000)
                await page.click(selector)
                await asyncio.sleep(2)
                current_url = page.url
                await browser.close()
                return f"✅ Clicked element '{selector}' successfully. Current URL: {current_url}"
        except Exception as e:
            return f"❌ Error while clicking element '{selector}': {str(e)}"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_click())
        finally:
            loop.close()
    except Exception as e:
        return f"❌ Failed to run click_element: {str(e)}"


@tool
def extract_links(url: str) -> str:
    """
    Extract all hyperlinks (anchor tags) from a webpage.
    Note: This tool may not work in cloud deployment environments without proper browser setup.

    Args:
        url (str): The webpage URL to extract links from.

    Returns:
        str: A formatted list of URLs found on the page.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "❌ Browser automation not available. Playwright is not installed."
    
    async def _extract():
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30000)
                anchors = await page.query_selector_all("a")
                links = []
                for a in anchors:
                    href = await a.get_attribute("href")
                    text = (await a.inner_text()).strip()
                    if href:
                        links.append(f"{text or 'No Text'}: {href}")
                await browser.close()
                if not links:
                    return f"No hyperlinks found on {url}"
                return f"🔗 Extracted {len(links)} links from {url}:\n\n" + "\n".join(links[:20])
        except Exception as e:
            return f"❌ Error extracting links from {url}: {str(e)}"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_extract())
        finally:
            loop.close()
    except Exception as e:
        return f"❌ Failed to run extract_links: {str(e)}"


@tool
def screenshot_page(url: str) -> str:
    """
    Take a screenshot of a webpage and save it as an image file.
    Returns the file path where the screenshot is saved so you can view it.
    Note: This tool may not work in cloud deployment environments without proper browser setup.

    Args:
        url (str): The webpage URL to screenshot.

    Returns:
        str: Path to the saved screenshot image file.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "❌ Browser automation not available. Playwright is not installed."
    
    async def _screenshot():
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30000)
                
                import time
                timestamp = int(time.time())
                screenshot_filename = f"screenshot_{timestamp}.png"
                screenshot_path = os.path.join(UPLOAD_DIR, screenshot_filename)
                
                await page.screenshot(path=screenshot_path, full_page=True)
                await browser.close()
                
                return f"📸 Screenshot captured successfully!\n\nSaved to: {screenshot_path}\n\nYou can view the screenshot at this location."
        except Exception as e:
            return f"❌ Error capturing screenshot: {str(e)}"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_screenshot())
        finally:
            loop.close()
    except Exception as e:
        return f"❌ Failed to run screenshot_page: {str(e)}"





@tool
def search_stack_overflow(query: str) -> str:
    """
    Intelligent Stack Overflow search tool.

    Purpose:
    - To find relevant Stack Overflow discussions about programming, firmware, or hardware issues.
    - To summarize useful content clearly, not just dump raw snippets.

    Behavior Guidelines (acts as system prompt for the LLM):
    - Use this tool ONLY for technical, programming, or debugging-related queries.
    - Always include question titles and URLs in the response.
    - Summarize the key solution or advice if available.
    - If no good results are found, respond politely stating that no relevant data was found.
    - Do NOT fabricate or guess answers — rely only on actual Stack Overflow results.

    Args:
        query (str): The user's question or problem description.

    Returns:
        str: A formatted list of top Stack Overflow results with links and short summaries.
    """
    search = DuckDuckGoSearchRun()
    results = search.run(f"site:stackoverflow.com {query}")
    return results


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
    search_stack_overflow,
    execute_sql_query,
    convert_units,
    browser_navigation,
    click_element,        
    extract_links,        
    screenshot_page       # 👈 new
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
