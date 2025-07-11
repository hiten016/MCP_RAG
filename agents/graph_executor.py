from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from agents.ingestion_agent import ingest_document
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import generate_llm_response

retriever = RetrievalAgent()

# Define state type
State = dict

# Node 1: Ingest uploaded file
def ingestion_node(state: State) -> State:
    file = state["file"]
    ingest_msg = ingest_document(file)
    state["chunks"] = ingest_msg.content["chunks"]
    return state

# Node 2: Retrieve relevant chunks
def retrieval_node(state: State) -> State:
    query = state["query"]
    chunks = state["chunks"]
    retriever.build_index(chunks)
    retrieval_msg = retriever.retrieve(query)
    state["top_chunks"] = retrieval_msg.content["relevant_chunks"]
    return state

# Node 3: LLM response generation
def llm_response_node(state: State) -> State:
    response_msg = generate_llm_response(state["query"], state["top_chunks"])
    state["answer"] = response_msg.content["response"]
    return state

# LangGraph Builder
def build_graph():
    builder = StateGraph(State)
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("llm_response", llm_response_node)

    builder.set_entry_point("ingestion")
    builder.add_edge("ingestion", "retrieval")
    builder.add_edge("retrieval", "llm_response")
    builder.add_edge("llm_response", END)

    return builder.compile()

# Entry point to run the graph
def execute_agent_graph(file, query):
    graph = build_graph()
    final_state = graph.invoke({"file": file, "query": query})
    return final_state["answer"]
