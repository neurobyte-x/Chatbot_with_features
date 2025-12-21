# 🤖 NeuroAgent - Multi-Tool Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent AI chatbot powered by **Google Gemini 2.5 Flash**, **LangChain**, and **LangGraph** with 12+ integrated tools including web search, code execution, browser automation, and AI vision capabilities.

![Demo Screenshot](https://via.placeholder.com/800x400?text=Add+Your+Screenshot+Here)

## ✨ Features

### 🔍 **Core Intelligence**
- **Web Search** - Real-time internet search via Google Serper API
- **Python Code Executor** - Safe execution of Python code snippets
- **Mathematical Calculator** - Advanced calculations with trig, log, and more
- **Stack Overflow Search** - Find coding solutions from Stack Overflow

### 🌐 **Browser Automation** (Powered by Playwright)
- **Navigate Webpages** - Extract text content from any URL
- **Click Elements** - Interact with webpage elements programmatically
- **Extract Links** - Get all hyperlinks from a webpage
- **Screenshot Pages** - Capture full-page screenshots

### 🖼️ **AI Vision** (Powered by Gemini 2.5 Pro)
- **Image Analysis** - Analyze screenshots, diagrams, code snippets
- **Visual Understanding** - Extract text and understand image content
- **Problem Recognition** - Identify issues in uploaded images

### 🛠️ **Additional Tools**
- **Weather Information** - Live weather data for any city
- **Unit Converter** - Temperature, length, weight conversions
- **SQL Query Executor** - Run SQL queries on SQLite databases

### 💬 **Smart Conversation**
- **Persistent Memory** - Thread-based conversation history
- **Auto-naming** - Intelligent chat session naming
- **Multi-turn Context** - Maintains context across conversations
- **Image Upload** - Analyze images directly in chat

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- API Keys:
  - [Google AI API Key](https://makersuite.google.com/app/apikey) (for Gemini)
  - [Serper API Key](https://serper.dev/) (for web search)
  - [OpenWeatherMap API Key](https://openweathermap.org/api) (for weather)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/neurobyte-x/NeuroAgent.git
   cd NeuroAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (for browser automation)
   ```bash
   playwright install chromium
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=chatbot-project
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_frontend.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 📖 Usage Examples

### Web Search
```
User: "What are the latest developments in AI?"
Bot: [Uses web_search tool and returns summarized results]
```

### Browser Automation
```
User: "Take a screenshot of https://leetcode.com"
Bot: [Captures screenshot and displays it]
```

### Image Analysis
```
User: [Uploads image] "What do you see in this image?"
Bot: [Analyzes image and describes content]
```

### Code Execution
```
User: "Execute this Python code: print([x**2 for x in range(10)])"
Bot: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Mathematical Calculations
```
User: "Calculate sin(45) + log10(100)"
Bot: 2.7071...
```

### Combined Operations
```
User: "Open https://example.com, take a screenshot, and analyze it"
Bot: [Executes multiple tools in sequence]
```

## 🏗️ Project Structure

```
NeuroAgent/
├── langgraph_backend.py      # Core backend logic with tools
├── streamlit_frontend.py      # Streamlit UI interface
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in git)
├── .env.example              # Example environment file
├── uploaded_images/          # Stored uploaded/screenshot images
├── __pycache__/              # Python cache
└── README.md                 # This file
```

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Google Gemini 2.5 Flash** | Primary LLM for fast responses |
| **Google Gemini 2.5 Pro** | Advanced vision and reasoning |
| **LangChain** | Tool integration framework |
| **LangGraph** | State-based workflow orchestration |
| **Streamlit** | Web UI framework |
| **Playwright** | Browser automation |
| **Google Serper API** | Web search |
| **OpenWeatherMap API** | Weather data |
| **LangSmith** | Observability and tracing |

## 🔧 Configuration

### Customizing Tools

Edit `langgraph_backend.py` to add or remove tools:

```python
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
    screenshot_page,
    # Add your custom tools here
]
```

### Adjusting LLM Model

Change the model in `langgraph_backend.py`:

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# or
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

## 📊 Available Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `web_search` | Search the web via Google Serper | "Search for Python tutorials" |
| `python_code_executor` | Execute Python code safely | "Run: print('Hello')" |
| `mathematical_calculator` | Perform math calculations | "Calculate sqrt(144)" |
| `weather_tool` | Get weather information | "Weather in Tokyo" |
| `image_reasoning_tool` | Analyze images with AI | Upload image + ask question |
| `browser_navigation` | Navigate and extract webpage content | "Go to example.com" |
| `click_element` | Click webpage elements | "Click the login button" |
| `extract_links` | Get all links from a page | "Extract links from github.com" |
| `screenshot_page` | Capture webpage screenshots | "Screenshot leetcode.com" |
| `search_stack_overflow` | Search Stack Overflow | "Find solution for React hooks" |
| `execute_sql_query` | Run SQL queries | "SELECT * FROM users" |
| `convert_units` | Convert between units | "Convert 100F to celsius" |

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add environment variables in Streamlit Cloud settings
5. Deploy!

### Deploy with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium
RUN playwright install-deps

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_frontend.py"]
```

Build and run:
```bash
docker build -t ai-chatbot .
docker run -p 8501:8501 --env-file .env ai-chatbot
```

## 🧪 Testing

### Test Browser Automation
```python
# In Python console
from langgraph_backend import screenshot_page

result = screenshot_page("https://example.com")
print(result)
```

### Test Image Analysis
```python
from langgraph_backend import image_reasoning_tool

analysis = image_reasoning_tool("path/to/image.png")
print(analysis)
```

## 🐛 Troubleshooting

### Playwright Issues
```bash
# Reinstall browsers
playwright install chromium

# Install system dependencies
playwright install-deps
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### API Key Issues
- Verify all API keys are in `.env`
- Check API key validity on respective platforms
- Ensure `.env` is in the project root

### Memory Issues
- Clear uploaded images folder periodically
- Restart the application
- Check system resources

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Test new tools thoroughly
- Update README with new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [Google](https://ai.google.dev/) for Gemini API
- [Streamlit](https://streamlit.io) for the beautiful UI framework
- [Playwright](https://playwright.dev/) for browser automation

## 📧 Contact

**Author**: neurobyte-x

- GitHub: [@neurobyte-x](https://github.com/neurobyte-x)
- Project Link: [https://github.com/neurobyte-x/Chatbot_with_features](https://github.com/neurobyte-x/NeuroAgent)

## 🗺️ Roadmap

- [ ] Add Composio integration for Gmail, Calendar, Slack
- [ ] Implement conversation export (PDF/JSON)
- [ ] Add voice input/output
- [ ] Support for multiple LLM providers
- [ ] Advanced image editing tools
- [ ] Plugin system for custom tools
- [ ] Multi-user support with authentication
- [ ] Mobile app version

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Made with ❤️ by neurobyte-x**
