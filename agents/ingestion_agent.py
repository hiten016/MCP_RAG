 
from utils.loaders import load_document
from utils.text_splitter import split_text
from agents.mcp import MCPMessage

def ingest_document(file) -> MCPMessage:
    text = load_document(file)
    chunks = split_text(text)
    return MCPMessage(type="INGEST", content={"chunks": chunks})
