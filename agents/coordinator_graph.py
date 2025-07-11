from langgraph.graph import StateGraph, END
from agents.mcp import MCPMessage

class CoordinatorGraphBuilder:
    def __init__(self, ingestion_agent, retrieval_agent, llm_agent):
        self.ingestion = ingestion_agent
        self.retrieval = retrieval_agent
        self.llm = llm_agent

    def build_graph(self):
        builder = StateGraph(dict)

        # --- Node: IngestionAgent
        def ingestion_node(state):
            print("🟢 [IngestionAgent] Node triggered")

            docs = state.get("docs")
            if not docs:
                raise ValueError("Missing 'docs' in state input")

            msg = MCPMessage(
                sender="CoordinatorGraph",
                receiver="IngestionAgent",
                msg_type="INGEST",
                payload={"files": docs}
            )

            chunks_msg = self.ingestion.handle(msg)
            print("🟢 [IngestionAgent] Output:", chunks_msg)

            new_state = state.copy()
            new_state["chunks_msg"] = chunks_msg
            return new_state

        # --- Node: RetrievalAgent
        def retrieval_node(state):
            print("🔵 [RetrievalAgent] Node triggered")

            chunks_msg = state.get("chunks_msg")
            query = state.get("query")

            if not chunks_msg or "chunks" not in chunks_msg["payload"]:
                raise ValueError("Missing 'chunks' in ingestion output")
            if not query:
                raise ValueError("Missing 'query' in state input")

            chunks = chunks_msg["payload"]["chunks"]

            # Step 1: Build the index
            index_msg = MCPMessage(
                sender="CoordinatorGraph",
                receiver="RetrievalAgent",
                msg_type="DOC_CHUNKS",
                payload={"chunks": chunks},
                trace_id=chunks_msg.get("trace_id")
            )
            self.retrieval.handle(index_msg)

            # Step 2: Run the query
            query_msg = MCPMessage(
                sender="CoordinatorGraph",
                receiver="RetrievalAgent",
                msg_type="QUERY",
                payload={"query": query},
                trace_id=chunks_msg.get("trace_id")
            )
            retrieval_response = self.retrieval.handle(query_msg)
            print("🔵 [RetrievalAgent] Output:", retrieval_response)

            new_state = state.copy()
            new_state["retrieval_msg"] = retrieval_response
            return new_state

        # --- Node: LLMResponseAgent
        def llm_node(state):
            print("🟣 [LLMResponseAgent] Node triggered")

            retrieval_msg = state.get("retrieval_msg")
            query = state.get("query")

            if not retrieval_msg or "top_chunks" not in retrieval_msg["payload"]:
                raise ValueError("Missing 'top_chunks' in retrieval output")
            if not query:
                raise ValueError("Missing 'query' in state input")

            top_chunks = retrieval_msg["payload"]["top_chunks"]

            msg = MCPMessage(
                sender="CoordinatorGraph",
                receiver="LLMResponseAgent",
                msg_type="ANSWER",
                payload={"top_chunks": top_chunks, "query": query},
                trace_id=retrieval_msg.get("trace_id")
            )

            llm_response = self.llm.handle(msg)
            print("🟣 [LLMResponseAgent] Output:", llm_response)

            new_state = state.copy()
            new_state["llm_msg"] = llm_response
            return new_state

        # --- Register Nodes
        builder.add_node("IngestionAgent", ingestion_node)
        builder.add_node("RetrievalAgent", retrieval_node)
        builder.add_node("LLMResponseAgent", llm_node)

        # --- Define Edges
        builder.set_entry_point("IngestionAgent")
        builder.add_edge("IngestionAgent", "RetrievalAgent")
        builder.add_edge("RetrievalAgent", "LLMResponseAgent")
        builder.add_edge("LLMResponseAgent", END)

        return builder.compile()
 
