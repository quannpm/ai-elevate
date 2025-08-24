import os
import json
from typing import Optional
from typing_extensions import TypedDict
from pathlib import Path

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
LLM_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-***")
LLM_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
EMBEDDING_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-***")
EMBEDDING_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# LangChain & community imports (as requested)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:
    # Fallback import path naming (some installs expose modules differently)
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
except Exception:
    InMemoryDocstore = None
    FAISS = None

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    # Some installs use langchain.schema or langchain.prompts
    try:
        from langchain.schema import Document
        from langchain.prompts import ChatPromptTemplate
    except Exception:
        Document = None
        ChatPromptTemplate = None

try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = None

# -----------------------------
# 1) TypedDict state for langgraph
# -----------------------------
class RAGState(TypedDict, total=False):
    question: str
    context: Optional[str]
    answer: Optional[str]


# -----------------------------
# 2) Retail documents (15 entries)
# -----------------------------
def get_retail_documents():
    texts = [
        "Walmart customers may return electronics within 30 days with a receipt and original packaging.",
        "Grocery items at Walmart can be returned within 90 days with proof of purchase, except perishable products.",
        "Walmart offers a 1-year warranty on most electronics and appliances. See product details for exceptions.",
        "Walmart Plus members get free shipping with no minimum order amount.",
        "Prescription medications purchased at Walmart are not eligible for return or exchange.",
        "Open-box items are eligible for return at Walmart within the standard return period, but must include all original accessories.",
        "If a Walmart customer does not have a receipt, most returns are eligible for store credit with valid photo identification.",
        "Walmart allows price matching for identical items found on Walmart.com and local competitor ads.",
        "Walmart Vision Center purchases may be returned or exchanged within 60 days with a receipt.",
        "Returns on cell phones at Walmart require the device to be unlocked and all personal data erased.",
        "Walmart gift cards cannot be redeemed for cash except where required by law.",
        "Seasonal merchandise at Walmart (e.g., holiday decorations) may have modified return windows, see in-store signage.",
        "Bicycles purchased at Walmart can be returned within 90 days if not used outdoors and with all accessories present.",
        "For online Walmart orders, customers can return items in store or by mail using the prepaid label.",
        "Walmart reserves the right to deny returns suspected of fraud or abuse.",
    ]
    docs = []
    for i, t in enumerate(texts, start=1):
        # Document expects page_content (some langchain variants use 'page_content')
        try:
            docs.append(Document(page_content=t, metadata={"source": f"doc_{i}", "id": i}))
        except Exception:
            # fallback if Document signature differs
            docs.append({"page_content": t, "metadata": {"source": f"doc_{i}", "id": i}})
    return docs


# -----------------------------
# 3) Build embeddings + FAISS vectorstore (in-memory)
# -----------------------------
def build_vectorstore(docs):
    if OpenAIEmbeddings is None or FAISS is None or InMemoryDocstore is None:
        print("Warning: langchain or langchain_community modules not available. Using mock retriever.")
        # Create a simple mock retriever that returns first 2 docs for any query
        class MockRetriever:
            def __init__(self, docs):
                self.docs = docs

            def get_relevant_documents(self, query, k=2):
                return self.docs[:k]

        return MockRetriever(docs)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_OPENAI_MODEL,
        openai_api_key=EMBEDDING_OPENAI_API_KEY,
        openai_api_base=OPENAI_ENDPOINT
    )

    # Build FAISS index from documents
    # FAISS.from_documents usually expects list[Document] and embeddings instance
    try:
        docstore = InMemoryDocstore({str(i): d for i, d in enumerate(docs)})
        vectorstore = FAISS.from_documents(docs, embeddings, docstore=docstore)
    except Exception as e:
        # If the exact signature differs, try without docstore
        vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


# -----------------------------
# 4) Chat model and prompt
# -----------------------------
def get_chat_model():
    if ChatOpenAI is None:
        print("Warning: ChatOpenAI not available. Using mock LLM.")
        class MockLLM:
            def __call__(self, prompt_text, *args, **kwargs):
                # Very simple echo-style mock
                return "MOCK ANSWER (LLM not available): I found information related to your query."

            # support legacy .invoke used in some examples
            def invoke(self, prompt):
                return type("R", (), {"content": "MOCK ANSWER (invoke): LLM not available."})

        return MockLLM()

    llm = ChatOpenAI(
        model_name=LLM_OPENAI_MODEL,
        openai_api_key=LLM_OPENAI_API_KEY,
        openai_api_base=OPENAI_ENDPOINT,
        temperature=LLM_OPENAI_TEMPERATURE
    )
    return llm


def build_prompt():
    # Use ChatPromptTemplate as requested
    if ChatPromptTemplate is None:
        # fallback to simple string template
        class SimplePrompt:
            def format(self, **kwargs):
                return f"{kwargs.get('context','')}\n\nUser question: {kwargs.get('question')}"
        return SimplePrompt()
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful Walmart support assistant. Use the provided information to answer product and policy questions. Cite retrieved info in your answer."),
                ("human", "{context}\n\nUser question: {question}"),
            ]
        )
        return prompt
    except Exception:
        # Some versions expect list of tuples in different forms
        class SimplePrompt:
            def format(self, **kwargs):
                return f"{kwargs.get('context','')}\n\nUser question: {kwargs.get('question')}"
        return SimplePrompt()


# -----------------------------
# 5) Node functions for StateGraph
# -----------------------------
def make_retrieve_node(retriever):
    def retrieve_node(state: RAGState) -> RAGState:
        q = state["question"]
        docs = retriever.invoke(q)
        # attempt to read .page_content or dict key
        pieces = []
        for d in docs:
            if hasattr(d, "page_content"):
                pieces.append(d.page_content)
            elif isinstance(d, dict) and "page_content" in d:
                pieces.append(d["page_content"])
            elif isinstance(d, str):
                pieces.append(d)
            else:
                # try metadata
                pieces.append(str(d))
        context = "\n".join(pieces)
        return {**state, "context": context}
    return retrieve_node


def make_generate_node(llm, prompt):
    def generate_node(state: RAGState) -> RAGState:
        # Format prompt
        try:
            formatted = prompt.format(context=state.get("context", ""), question=state["question"])
        except Exception:
            # If ChatPromptTemplate gives a PromptValue with to_messages, try to convert
            try:
                pv = prompt.format_prompt(context=state.get("context", ""), question=state["question"])
                # Some PromptValue exposes to_messages()
                formatted = pv.to_string() if hasattr(pv, "to_string") else str(pv)
            except Exception:
                formatted = f"{state.get('context','')}\n\nUser question: {state['question']}"

        # Call the LLM. Try a few common interfaces for different langchain versions.
        answer_text = None
        try:
            # Some ChatOpenAI implementations accept a plain string call
            res = llm.invoke(formatted)  # type: ignore
            # If llm returns an object or string
            if isinstance(res, str):
                answer_text = res
            elif hasattr(res, "content"):
                answer_text = res.content
            elif hasattr(res, "generations"):
                # llm.generate returns generations list
                try:
                    answer_text = res.generations[0][0].text
                except Exception:
                    answer_text = str(res)
            elif hasattr(res, "choices"):
                # openai-like response
                try:
                    answer_text = res.choices[0].message.content
                except Exception:
                    answer_text = str(res)
            else:
                answer_text = str(res)
        except Exception:
            try:
                inv = llm.invoke(formatted)  # type: ignore
                answer_text = getattr(inv, "content", str(inv))
            except Exception:
                # last resort: mock reply combining context and question
                answer_text = (
                    "I could not contact the LLM. Based on retrieved context:\n\n"
                    + (state.get("context") or "")
                    + f"\n\nSuggested answer to: {state['question']}"
                )
        return {**state, "answer": answer_text}
    return generate_node


# -----------------------------
# 6) Main flow: build graph, run demos, produce outputs & README
# -----------------------------
def main():
    out_dir = Path("./tempp")
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = get_retail_documents()
    retriever = build_vectorstore(docs)
    llm = get_chat_model()
    prompt = build_prompt()

    # Build LangGraph StateGraph
    if StateGraph is None:
        print("Warning: langgraph not available. Running nodes sequentially without StateGraph.")
        # Define simple sequential runner
        def run_once(question):
            state = RAGState(question=question)
            retrieve = make_retrieve_node(retriever)
            gen = make_generate_node(llm, prompt)
            state = retrieve(state)
            state = gen(state)
            return state
        graph_invoke = run_once
    else:
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", make_retrieve_node(retriever))
        builder.add_node("generate", make_generate_node(llm, prompt))
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.set_finish_point("generate")
        rag_graph = builder.compile()
        def graph_invoke(state):
            return rag_graph.invoke(state)

    # Dummy auto questions (no input())
    demo_questions = [
        "Can I return a Walmart bicycle if I've ridden it outdoors?",
        "What is the return window for electronics purchased at Walmart?",
        "Can I return prescription medications?",
        "How does Walmart handle returns without a receipt?",
    ]

    results = []
    for q in demo_questions:
        state_in = RAGState(question=q)
        result = graph_invoke(state_in)
        results.append({"question": q, "context": result.get("context"), "answer": result.get("answer")})

    # Save results as JSON and print a demo
    results_path = out_dir / "assignment15_demo_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Demo Q&A outputs (first demo shown):\n")
    if results:
        first = results[0]
        print("Question:", first["question"])
        print("\nRetrieved Context:\n", first["context"])
        print("\nGenerated Answer:\n", first["answer"])
    else:
        print("No results generated.")

    print(f"\nSaved results to: {results_path}")

if __name__ == '__main__':
    main()
