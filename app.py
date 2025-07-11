from agents.coordinator_graph import CoordinatorGraphBuilder 
from agents.ingestion_agent import IngestionAgent 
from agents.retrieval_agent import RetrievalAgent 
from agents.llm_response_agent import LLMResponseAgent 
from agents.mcp import MCPMessage 
 
ingestion_agent = IngestionAgent() 
retrieval_agent = RetrievalAgent() 
llm_agent = LLMResponseAgent() 
 
graph = CoordinatorGraphBuilder(ingestion_agent, retrieval_agent, llm_agent).build_graph() 
 
if __name__ == "__main__": 
    example_files = ["temp/sample.pdf"] 
    query = "What is the summary of the document?" 
    msg = MCPMessage.make_user_query(docs=example_files, query=query) 
    final_state = graph.invoke(msg) 
    print("\n? Final Response:", final_state.payload.get("response")) 
