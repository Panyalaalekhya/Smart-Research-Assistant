from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
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

# Tools
search_tool = TavilySearchResults(max_results=3)

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Node 1 - Search the web
def search_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    print(f"\nSearching for: {query}")
    results = search_tool.invoke(query +  " detailed explanation 2026")
    sources = [r["url"] for r in results]
    state["sources"] = sources
    research_data = "\n".join([r["content"] for r in results])
    state["research_data"] = research_data
    state["attempts"] = state.get("attempts", 0) + 1
    return state

# Node 2 - Generate answer
def answer_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    research_data = state["research_data"]
    prompt = f"Based on this research:\n{research_data}\n\nAnswer this query: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\nAnswer: {response.content}")
    print("\nSources:")
    for source in state["sources"]:
        print(f"  - {source}")
    state["messages"].append(AIMessage(content=response.content))
    return state

# Node 3 - Reflect on answer
def reflection_node(state: ResearchState) -> ResearchState:
    answer = state["messages"][-1].content
    prompt = f"Is this answer detailed and complete? Reply ONLY with 'yes' or 'no', nothing else:\n{answer}"
    response = llm.invoke([HumanMessage(content=prompt)])
    reflection = response.content.strip().lower()
    state["reflection"] = reflection
    state["is_complete"] = "yes" in reflection
    print(f"\nReflection: {reflection}")
    return state

# Conditional edge
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

# Run
query = input("Enter your research query: ")
result = agent.invoke({
    "messages": [],
    "query": query,
    "research_data": "",
    "reflection": "",
    "is_complete": False,
    "attempts": 0,
    "sources": [],
})