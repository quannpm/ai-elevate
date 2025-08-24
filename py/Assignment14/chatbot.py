import os
import json
from typing import List, Dict, Any

# -----------------------------
# Optional imports (graceful)
# -----------------------------
OPENAI_AVAILABLE = False
LANGGRAPH_AVAILABLE = False
TOOLS_DECORATOR_AVAILABLE = False
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY", "tvly-dev-***")

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
LLM_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-***")
LLM_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
EMBEDDING_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-***")
EMBEDDING_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

try:
    # LangChain wrappers
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        # langchain_openai integration
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False
except Exception:
    # If Document / FAISS not available, we'll use simple structures for mock chunks / retriever
    Document = None

# Try to import LangGraph and related helpers
try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# Tool decorator (LangChain's @tool)
try:
    # preferred import path in recent docs
    from langchain_core.tools import tool
    TOOLS_DECORATOR_AVAILABLE = True
except Exception:
    try:
        from langchain.tools import tool
        TOOLS_DECORATOR_AVAILABLE = True
    except Exception:
        TOOLS_DECORATOR_AVAILABLE = False

# Tavily search tool (optional)
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

# -----------------------------
# Mock knowledge base
# -----------------------------
mock_chunks = [
    Document(page_content="Patients with a sore throat should drink warm fluids and avoid cold beverages.")
    if Document else {"page_content": "Patients with a sore throat should drink warm fluids and avoid cold beverages."},
    Document(page_content="Mild fevers under 38.5°C can often be managed with rest and hydration.")
    if Document else {"page_content": "Mild fevers under 38.5°C can often be managed with rest and hydration."},
    Document(page_content="If a patient reports dizziness, advise checking their blood pressure and hydration level.")
    if Document else {"page_content": "If a patient reports dizziness, advise checking their blood pressure and hydration level."},
    Document(page_content="Persistent coughs lasting more than 2 weeks should be evaluated for infections or allergies.")
    if Document else {"page_content": "Persistent coughs lasting more than 2 weeks should be evaluated for infections or allergies."},
    Document(page_content="Patients experiencing fatigue should consider iron deficiency or poor sleep as potential causes.")
    if Document else {"page_content": "Patients experiencing fatigue should consider iron deficiency or poor sleep as potential causes."},
]

# Dummy inputs (auto inputs) — final checklist requirement (no input())
AUTO_INPUTS = [
    {"name": "Alice Nguyen", "age": 29, "symptoms": "sore throat, mild fever, tiredness"},
    {"name": "Mr. Tran",    "age": 67, "symptoms": "dizziness, lightheaded when standing up"},
    {"name": "Linh",        "age": 15, "symptoms": "persistent cough for 3 weeks"}
]

# -----------------------------
# Retriever setup (FAISS if available, else fallback)
# -----------------------------
def build_retriever():
    # Try to build FAISS retriever with embeddings
    if OPENAI_AVAILABLE:
        try:
            embed_model = OpenAIEmbeddings(
                model=EMBEDDING_OPENAI_MODEL,
                openai_api_key=EMBEDDING_OPENAI_API_KEY,
                openai_api_base=OPENAI_ENDPOINT
            )
            db = FAISS.from_documents(mock_chunks, embed_model)
            retriever = db.as_retriever()
            print("[INFO] Built FAISS retriever with embeddings.")
            return retriever
        except Exception as e:
            print(f"[WARN] Couldn't build FAISS/embeddings: {e}")
    # Fallback simple retriever
    def simple_retriever(query: str) -> List[str]:
        q = query.lower()
        results = []
        for d in mock_chunks:
            content = d.page_content if hasattr(d, "page_content") else d["page_content"]
            if any(tok in content.lower() for tok in q.split()):
                results.append(content)
        if not results:
            results = [d.page_content if hasattr(d, "page_content") else d["page_content"] for d in mock_chunks]
        return results
    print("[INFO] Using fallback simple retriever.")
    return simple_retriever

# -----------------------------
# Tool definitions
# -----------------------------

def maybe_tool(fn):
    if TOOLS_DECORATOR_AVAILABLE and callable(tool):
        return tool(fn)
    else:
        # identity (no-op) if decorator isn't available
        return fn

@maybe_tool
def retrieve_advice(user_input: str) -> str:
    """Searches internal documents for relevant patient advice (tool for model to call)."""
    retriever = build_retriever()
    try:
        # If this retriever is a LangChain retriever object
        docs = retriever.invoke(user_input)  # type: ignore[attr-defined]
        return "\\n".join(doc.page_content for doc in docs)
    except Exception:
        # assume callable simple_retriever
        docs = retriever(user_input)
        return "\\n".join(d for d in docs)

# Tavily tool (optional). If tavily not present, make a dummy tool to keep API surface identical.
if TAVILY_AVAILABLE:
    tavily_tool = TavilySearch(tavily_api_key="your_api_key_here", k=3)
else:
    @maybe_tool
    def tavily_tool(query: str) -> str:
        """Dummy Tavily fallback - returns an empty or placeholder string."""
        return "[Tavily unavailable in this environment]"

# -----------------------------
# LLM setup
# -----------------------------

class MockLLMWithTools:
    """Simple mock that mimics a model bound with tools for offline testing."""
    def __init__(self):
        self._tools = {}
    def bind_tools(self, tools: List[Any]):
        # map tool name -> callable (work with either decorated tools or plain funcs)
        for t in tools:
            # the decorator produces a wrapper with a .name or uses function.__name__
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            self._tools[name] = t
        return self
    def invoke(self, messages: List[Any]):
        # Very simple behaviour: look at last human message and return a short reply.
        last = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        # if the model were to call a tool, we'll mimic no tool call
        class Resp:
            def __init__(self, text):
                self.content = text
                self.tool_calls = []
        # produce a response using simple heuristics
        text = "Preliminary advice: rest, hydrate, monitor symptoms. Use retrieve_advice for internal guidance."
        if "dizzy" in last.lower() or "dizziness" in last.lower():
            text = "Preliminary advice: sit/lie down, check hydration and blood pressure. Seek urgent care for fainting."
        if "sore throat" in last.lower():
            text = "Warm fluids, rest, lozenges; seek care for severe pain or high fever."
        if "cough" in last.lower():
            text = "Monitor cough; if >2 weeks or difficulty breathing, see a clinician."
        return Resp(text)

def create_llm_with_tools():
    if OPENAI_AVAILABLE:
        try:
            llm = ChatOpenAI(
                model_name=LLM_OPENAI_MODEL,
                openai_api_key=LLM_OPENAI_API_KEY,
                openai_api_base=OPENAI_ENDPOINT,
                temperature=LLM_OPENAI_TEMPERATURE
            )
            # If the model supports tool binding (models supporting tool calling), bind tools
            try:
                llm_with_tools = llm.bind_tools([retrieve_advice, tavily_tool])
                print("[INFO] ChatOpenAI instantiated and tools bound.")
                return llm_with_tools
            except Exception:
                print("[WARN] ChatOpenAI instantiated but couldn't bind tools; returning model without bound tools.")
                return llm
        except Exception as e:
            print(f"[WARN] Couldn't instantiate ChatOpenAI: {e}")
    # Fallback to mock
    print("[INFO] Using MockLLMWithTools fallback.")
    mock = MockLLMWithTools()
    mock.bind_tools([retrieve_advice, tavily_tool])
    return mock

# -----------------------------
# LangGraph nodes & graph
# -----------------------------
def build_graph(llm_with_tools):
    if not LANGGRAPH_AVAILABLE:
        return None

    # Model-calling node
    def call_model(state: MessagesState):
        messages = state["messages"]  # list of message objects
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Decide whether to route to tools node
    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        # If the last message includes tool_calls (LangChain message with tool_calls), go to tools node
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    # Build graph
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("call_model", call_model)
    # ToolNode wraps tools so the graph can call them
    try:
        node_tools = ToolNode([retrieve_advice, tavily_tool])
        graph_builder.add_node("tools", node_tools)
    except Exception:
        # If ToolNode isn't present or fails, skip creating explicit tools node.
        pass

    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "call_model")

    graph = graph_builder.compile()
    return graph

# -----------------------------
# Execution / Runner
# -----------------------------
def main():
    print("=== Assignment 14 Chatbot Agent ===")
    retriever = build_retriever()  # not used directly by langgraph but available for the tool
    llm_with_tools = create_llm_with_tools()

    if LANGGRAPH_AVAILABLE:
        graph = build_graph(llm_with_tools)
        try:
            example_state = {
                "messages": [
                    SystemMessage(content="You are a helpful medical assistant. Use tools if needed."),
                    HumanMessage(content="I feel tired and have a sore throat. What should I do?")
                ]
            }
            result = graph.invoke(example_state)
            print("\\nFinal Response (Graph):")
            # The result is a dict with "messages": [ ... ]
            final_msg = result["messages"][-1]
            print(getattr(final_msg, "content", str(final_msg)))
        except Exception as e:
            print(f"[WARN] Graph invocation failed: {e}")
    else:
        print("[WARN] LangGraph not available — running a fallback sequential flow.\\n")
        outputs = []
        for p in AUTO_INPUTS:
            messages = [
                {                    "role": "system",                   "content": "You are a helpful medical assistant. Use tools if needed."              },
                {                    "role": "user",                    "content": f"Name: {p['name']}; Age: {p['age']}; Symptoms: {p['symptoms']}"             }
            ]
            # The mock llm_with_tools provides invoke(messages) in fallback case
            try:
                resp = llm_with_tools.invoke(type('M', (), {'content': messages[-1]['content']}))  # lightweight
                text = getattr(resp, 'content', str(resp))
            except Exception as e:
                text = f"[ERROR generating response]: {e}"
            # also gather internal advice via retrieve_advice tool
            internal = retrieve_advice(p['symptoms'])
            outputs.append({"patient": p, "response": text, "internal_advice": internal})
            print("\\n---")
            print(f"Patient: {p['name']} (Age {p['age']})")
            print("Symptoms:", p['symptoms'])
            print("\\nAgent Advice:\\n", text)
            print("\\nInternal Knowledge (retriever):\\n", internal)
        # Save sample outputs to file (final checklist)
        with open("assignment14_sample_outputs.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print("\\n[INFO] Sample outputs saved to assignment14_sample_outputs.json")

if __name__ == '__main__':
    main()
