"""
AI Weather & Web Search (Manual Routing, No Function Calls)

Motivation:
- Proxy model (GPT-5) rejects: temperature≠default, stop param, function-role messages.
- Removed LangChain agent. Implemented manual lightweight router.

Features:
- Route query to weather or web search.
- Weather: OpenWeatherMap (pyowm wrapper) OR (fallback simple parse).
- Search: Tavily results + LLM summarization.
- Conversation memory (simple list) for context (optional).

Env (.env):
  OPENAI_API_KEY=...
  OPENAI_MODEL=GPT-5
  OPENAI_BASE_URL=https://proxy.example.com/use   (optional; /v1 auto-added)
  OPENWEATHERMAP_API_KEY=...
  TAVILY_API_KEY=...

Install:
  pip install langchain-openai langchain-community tavily-python python-dotenv openai tiktoken pyowm
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------------
# Load .env explicitly
# -------------------------
SCRIPT_DIR = Path(__file__).parent
dotenv_path = SCRIPT_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# -------------------------
# Read env
# -------------------------
api_key = os.getenv("OPENAI_API_KEY")
model_name = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
base_url_raw = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
owm_key = os.getenv("OPENWEATHERMAP_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

missing = [v for v, val in [
    ("OPENAI_API_KEY", api_key),
    ("OPENWEATHERMAP_API_KEY", owm_key),
    ("TAVILY_API_KEY", tavily_key),
] if not val]
if missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

# Normalize base URL (/v1)
openai_api_base = ""
if base_url_raw:
    openai_api_base = base_url_raw if base_url_raw.endswith("/v1") else base_url_raw + "/v1"

# LLM kwargs (NO temperature, NO stop)
llm_kwargs = {
    "model": model_name,
    "openai_api_key": api_key,
    "timeout": 40
}
if openai_api_base:
    llm_kwargs["openai_api_base"] = openai_api_base

print(f"[INFO] Model: {model_name}")
print(f"[INFO] Base URL: {openai_api_base or 'OpenAI default'}")
print("[INFO] Initializing LLM...")

llm = ChatOpenAI(**llm_kwargs)

# Tools (direct, not via agent)
weather_api = OpenWeatherMapAPIWrapper()
tavily_tool = TavilySearchResults(max_results=5)

# Simple in-memory chat history
CHAT_HISTORY: List[Dict[str, str]] = []  # each: {"role": "user"|"assistant", "content": "..."}

# -------------------------
# Routing Heuristics
# -------------------------
WEATHER_KEYWORDS = re.compile(r"\b(weather|temperature|forecast|rain|snow|humidity)\b", re.IGNORECASE)

def heuristic_route(query: str) -> str:
    if WEATHER_KEYWORDS.search(query):
        return "weather"
    return "search"

ROUTE_CLASSIFIER_SYSTEM = """You are a classifier.
Decide route for a user query. Return ONLY one token: WEATHER or SEARCH.
Criteria:
- WEATHER: question about current weather, temperature, forecast in a location.
- SEARCH: anything else (news, trends, general info).
Output exactly WEATHER or SEARCH.
"""

def llm_route(query: str) -> str:
    """Fallback LLM router (only if heuristic uncertain)."""
    messages = [
        {"role": "system", "content": ROUTE_CLASSIFIER_SYSTEM},
        {"role": "user", "content": query}
    ]
    try:
        resp = llm.invoke(messages)
        txt = resp.content.strip().upper()
        if "WEATHER" in txt[:10]:
            return "weather"
        return "search"
    except Exception:
        return "search"

def extract_city(query: str) -> str:
    """
    Naive city extraction:
    - Look for patterns: 'weather in CITY'
    - Else return everything after 'in ' if present
    Fallback: return original query (api may fail then).
    """
    m = re.search(r"weather\s+(in|at)\s+([A-Za-z ,'-]+)", query, re.IGNORECASE)
    if m:
        return m.group(2).strip()
    m2 = re.search(r"in\s+([A-Za-z ,'-]+)$", query, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    # Fallback: first comma segment
    return query.split("?")[0].strip()

def call_weather(query: str) -> str:
    city = extract_city(query)
    try:
        result = weather_api.run(city)
        return f"Weather data for {city}:\n{result}"
    except Exception as e:
        return f"Weather fetch failed ({city}): {e}"

def call_search(query: str) -> str:
    try:
        raw = tavily_tool.invoke({"query": query})
    except Exception as e:
        return f"Search failed: {e}"
    # raw is list[dict] usually
    docs = []
    if isinstance(raw, list):
        for item in raw:
            title = item.get("title") or ""
            content = item.get("content") or ""
            url = item.get("url") or ""
            docs.append(f"- {title}\n  {content}\n  {url}")
    else:
        docs.append(str(raw))
    joined = "\n".join(docs[:5])
    # Summarize with LLM for final answer (no tools)
    system_summary = "You summarize search findings into a concise, helpful answer. Cite URLs briefly."
    messages = [
        {"role": "system", "content": system_summary},
        {"role": "user", "content": f"Query: {query}\nResults:\n{joined}"}
    ]
    try:
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        return f"Summary failed: {e}\nRaw:\n{joined}"

def answer(query: str) -> str:
    # Append user to history (optional future enhancement: send part of history)
    CHAT_HISTORY.append({"role": "user", "content": query})

    route = heuristic_route(query)
    # (Optional) If heuristic ambiguous, could use llm_route(query)
    if route not in ("weather", "search"):
        route = llm_route(query)

    if route == "weather":
        result = call_weather(query)
    else:
        result = call_search(query)

    CHAT_HISTORY.append({"role": "assistant", "content": result})
    return result

def run_samples():
    samples = [
        "What is the weather in Birmingham, UK today?",
        "Search latest news about AI in healthcare.",
        "What's the weather in Birmingham, UK and give clothing advice.",
        "Tell me temperature in Birmingham, UK now.",
        "Find recent breakthroughs in quantum computing."
    ]
    for q in samples:
        print(f"\n=== Query: {q} ===")
        print(answer(q))

def interactive_loop():
    print("Type 'exit' to quit.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "exit":
            print("Goodbye.")
            break
        print(answer(q))

if __name__ == "__main__":
    run_samples()
    # interactive_loop()