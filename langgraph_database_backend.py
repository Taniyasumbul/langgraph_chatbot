from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from datetime import datetime
import json

load_dotenv()

# ================================ TOOLS ================================

@tool
def web_scraper(url: str) -> str:
    """Scrape and extract text content from a webpage. Use this to read articles, blogs, or documentation."""
    try:
        from bs4 import BeautifulSoup
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit to first 2000 characters
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        return f"Scraping failed: {str(e)}"

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information on any topic. More reliable than web search for factual information."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'extract' in data:
            return f"Title: {data['title']}\n\n{data['extract']}\n\nRead more: {data['content_urls']['desktop']['page']}"
        else:
            return "No Wikipedia article found for this query."
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

@tool
def analyze_json(json_string: str) -> str:
    """Parse and analyze JSON data. Provides insights like key counts, data types, and structure."""
    try:
        data = json.loads(json_string)
        
        def analyze(obj, depth=0):
            if isinstance(obj, dict):
                return f"Object with {len(obj)} keys: {list(obj.keys())[:5]}"
            elif isinstance(obj, list):
                return f"Array with {len(obj)} items"
            else:
                return f"{type(obj).__name__}: {str(obj)[:50]}"
        
        analysis = analyze(data)
        return f"JSON Analysis:\n{analysis}\n\nFull structure:\n{json.dumps(data, indent=2)[:500]}"
    except Exception as e:
        return f"JSON parsing failed: {str(e)}"

@tool
def file_analyzer(file_content: str, file_type: str) -> str:
    """Analyze file content. Supports text, CSV, JSON. Returns statistics and insights."""
    try:
        if file_type == 'csv':
            lines = file_content.split('\n')
            rows = len(lines)
            cols = len(lines[0].split(',')) if lines else 0
            return f"CSV Analysis:\n- Rows: {rows}\n- Columns: {cols}\n- Preview:\n{file_content[:200]}"
        
        elif file_type == 'json':
            data = json.loads(file_content)
            return analyze_json(file_content)
        
        else:  # text
            words = len(file_content.split())
            chars = len(file_content)
            lines = len(file_content.split('\n'))
            return f"Text Analysis:\n- Words: {words}\n- Characters: {chars}\n- Lines: {lines}\n- Preview:\n{file_content[:300]}"
    except Exception as e:
        return f"File analysis failed: {str(e)}"

@tool
def data_processor(data: str, operation: str) -> str:
    """Process data with operations: sort, filter, aggregate, transform. Input should be comma-separated values."""
    try:
        items = [x.strip() for x in data.split(',')]
        
        if operation == 'sort':
            try:
                return ', '.join(sorted(items, key=float))
            except:
                return ', '.join(sorted(items))
        
        elif operation == 'sum':
            numbers = [float(x) for x in items if x.replace('.','').replace('-','').isdigit()]
            return f"Sum: {sum(numbers)}"
        
        elif operation == 'average':
            numbers = [float(x) for x in items if x.replace('.','').replace('-','').isdigit()]
            return f"Average: {sum(numbers)/len(numbers) if numbers else 0}"
        
        elif operation == 'unique':
            return ', '.join(set(items))
        
        elif operation == 'count':
            return f"Count: {len(items)}"
        
        else:
            return "Supported operations: sort, sum, average, unique, count"
    except Exception as e:
        return f"Data processing failed: {str(e)}"

@tool
def create_chart_data(data_description: str) -> str:
    """Generate chart-ready data based on description. Returns JSON format for visualization."""
    try:
        import random
        
        if 'bar' in data_description.lower():
            categories = ['A', 'B', 'C', 'D', 'E']
            values = [random.randint(10, 100) for _ in range(5)]
            result = {'type': 'bar', 'categories': categories, 'values': values}
        
        elif 'line' in data_description.lower():
            points = [{'x': i, 'y': random.randint(10, 100)} for i in range(10)]
            result = {'type': 'line', 'data': points}
        
        else:
            result = {'type': 'data', 'message': 'Specify chart type: bar or line'}
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Chart generation failed: {str(e)}"

@tool
def api_caller(url: str, method: str = "GET") -> str:
    """Call any public API and return the response. Supports GET requests."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
            return f"Status: {response.status_code}\n\nResponse:\n{response.text[:1000]}"
        else:
            return "Only GET method is supported for safety reasons."
    except Exception as e:
        return f"API call failed: {str(e)}"

@tool
def text_analyzer(text: str) -> str:
    """Analyze text for sentiment, word frequency, and readability metrics."""
    try:
        words = text.lower().split()
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'worst', 'horrible']
        
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        sentiment = "Positive" if pos_count > neg_count else "Negative" if neg_count > pos_count else "Neutral"
        
        return f"""Text Analysis:
- Total Words: {word_count}
- Unique Words: {unique_words}
- Average Word Length: {avg_word_length:.2f}
- Sentiment: {sentiment} (Positive: {pos_count}, Negative: {neg_count})
- Vocabulary Richness: {(unique_words/word_count*100):.1f}%"""
    except Exception as e:
        return f"Text analysis failed: {str(e)}"

@tool
def regex_matcher(text: str, pattern: str) -> str:
    """Find patterns in text using regex. Extract emails, phone numbers, URLs, etc."""
    try:
        import re
        
        matches = re.findall(pattern, text)
        
        if matches:
            return f"Found {len(matches)} matches:\n" + "\n".join(str(m) for m in matches[:20])
        else:
            return "No matches found for the given pattern."
    except Exception as e:
        return f"Regex matching failed: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid mathematical expression."""
    try:
        # Safe eval for math operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def python_code_executor(code: str) -> str:
    """Execute simple Python code snippets. Only use for safe, simple calculations or data processing."""
    try:
        # Very restricted execution environment
        allowed_builtins = {
            'print': print, 'len': len, 'range': range, 'str': str,
            'int': int, 'float': float, 'list': list, 'dict': dict,
            'sum': sum, 'min': min, 'max': max, 'abs': abs
        }
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec(code, {"__builtins__": allowed_builtins}, {})
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        return f"Execution error: {str(e)}"

# Tool list
tools = [
    search_wikipedia,
    web_scraper,
    calculate,
    get_current_time,
    python_code_executor,
    analyze_json,
    file_analyzer,
    data_processor,
    create_chart_data,
    api_caller,
    text_analyzer,
    regex_matcher
]

# ================================ LLM WITH TOOLS ================================

# Initialize Gemini LLM with tools
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-exp",
    temperature=0.7,
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ================================ STATE ================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================================ AGENT NODES ================================

def agent_node(state: AgentState):
    """Main agent node that decides whether to use tools or respond directly."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should use tools or end the conversation."""
    last_message = state['messages'][-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return "end"

# Create tool node
tool_node = ToolNode(tools)

# ================================ GRAPH SETUP ================================

# Create SQLite connection and checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

# Compile with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

# ================================ UTILITY FUNCTIONS ================================

def retrieve_all_threads():
    """Retrieve all unique thread IDs from the database."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get('configurable', {}).get('thread_id')
            if thread_id:
                all_threads.add(thread_id)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    
    return list(all_threads)

def get_available_tools():
    """Return information about available tools."""
    tool_info = []
    for tool in tools:
        tool_info.append({
            'name': tool.name,
            'description': tool.description
        })
    return tool_info