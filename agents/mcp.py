class MCPMessage:
    def __init__(self, type: str, content: dict):
        self.type = type  # e.g., "INGEST", "QUERY", "ANSWER"
        self.content = content

    def __repr__(self):
        return f"<MCPMessage type={self.type}, content_keys={list(self.content.keys())}>"
 
