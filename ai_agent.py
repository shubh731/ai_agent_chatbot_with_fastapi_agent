from dotenv import load_dotenv
load_dotenv()
import os
from typing import List, TypedDict

# Step 1: API Key Setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step 2: LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# Define tools
search_tool = TavilySearchResults(max_results=2)

# --------------------------
# MULTI-AGENT SETUP SECTION
# --------------------------

class AgentState(TypedDict):
    query: str
    result: str

# Example single node "agent" - simulate or route to LLM
def simple_agent_executor(state: AgentState) -> AgentState:
    query = state["query"]
    # Here you can call your LLM (currently just simulating)
    result = f"🧠 Multi-Agent: Processed response for: {query}"
    return {"query": query, "result": result}

def build_multi_agent_graph(llm, allow_search=False):
    graph = StateGraph(state_schema=AgentState)
    graph.add_node("agent", RunnableLambda(simple_agent_executor))
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph.compile()

# --------------------------
# MAIN ENTRY FUNCTION
# --------------------------

def get_response_from_ai_agent(llm_id, query: List[str], allow_search: bool, system_prompt: str, provider: str, use_multi_agent: bool = False):
    # Select LLM
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError("Unsupported provider")

    # MULTI-AGENT PATH
    if use_multi_agent:
        graph = build_multi_agent_graph(llm, allow_search)
        result = graph.invoke({"query": query[0], "result": ""})
        return result["result"]

    # SINGLE AGENT PATH
    tools = [search_tool] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]
