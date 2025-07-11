from sentence_transformers import SentenceTransformer
import faiss
from agents.mcp import MCPMessage

model = SentenceTransformer('all-MiniLM-L6-v2')

class RetrievalAgent:
    def __init__(self):
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        embeddings = model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.text_chunks = chunks

    def retrieve(self, query, top_k=5):
        query_emb = model.encode([query])
        _, indices = self.index.search(query_emb, top_k)
        top_chunks = [self.text_chunks[i] for i in indices[0]]
        return MCPMessage(type="QUERY", content={"relevant_chunks": top_chunks})
 
