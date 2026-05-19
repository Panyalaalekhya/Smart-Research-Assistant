import streamlit as st
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🔍",
    layout="wide"
)
import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

# State
class ResearchState(TypedDict):
    messages: List
    query: str
    research_data: str
    reflection: str
    is_complete: bool
    attempts: int
    sources: List

# Tools and LLM
search_tool = TavilySearchResults(max_results=3)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Nodes
def search_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    results = search_tool.invoke(query + " detailed explanation 2026")
    research_data = "\n".join([r["content"] for r in results])
    sources = [r["url"] for r in results]
    state["research_data"] = research_data
    state["sources"] = sources
    state["attempts"] = state.get("attempts", 0) + 1
    return state

def answer_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    research_data = state["research_data"]
    prompt = f"Based on this research:\n{research_data}\n\nAnswer this query: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(AIMessage(content=response.content))
    return state

def reflection_node(state: ResearchState) -> ResearchState:
    answer = state["messages"][-1].content
    prompt = f"Is this answer detailed and complete? Reply ONLY with 'yes' or 'no', nothing else:\n{answer}"
    response = llm.invoke([HumanMessage(content=prompt)])
    reflection = response.content.strip().lower()
    state["reflection"] = reflection
    state["is_complete"] = "yes" in reflection
    return state

def should_continue(state: ResearchState) -> str:
    if state["is_complete"] or state["attempts"] >= 2:
        return "end"
    else:
        return "search"

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)
graph.add_node("reflection", reflection_node)
graph.add_edge(START, "search")
graph.add_edge("search", "answer")
graph.add_edge("answer", "reflection")
graph.add_conditional_edges("reflection", should_continue, {
    "end": END,
    "search": "search"
})
agent = graph.compile()

# line 80
history_file = "search_history.json"

def load_history():
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(history_file, "w") as f:
        json.dump(history, f)

# Streamlit UI
with st.sidebar:
    st.header("About")
    st.write("AI Research Assistant that searches web and reflects on answers.")
    st.write("**Built with:**")
    st.write("- LangGraph")
    st.write("- Groq AI")
    st.write("- Tavily Search")
    st.write("- Streamlit")
    st.divider()
    st.header("Search History")
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    for item in st.session_state.history:
        st.write(f"- {item}")

st.title("Smart Research Assistant")
st.write("Ask any question and I will research it for you!")

query = st.text_input("Enter your research query:")

if st.button("Research"):
    if query:
        with st.spinner("Researching..."):
            result = agent.invoke({
                "messages": [],
                "query": query,
                "research_data": "",
                "reflection": "",
                "is_complete": False,
                "attempts": 0,
                "sources": []
            })
            st.session_state.history.append(query)
            save_history(st.session_state.history)
        st.subheader("Answer:")
        st.write(result["messages"][-1].content)
        st.divider()
        st.subheader("Sources:")
        for source in result["sources"]:
            st.write(f"- {source}")
        st.divider()
        st.subheader("Reflection:")
        st.write(result["reflection"])