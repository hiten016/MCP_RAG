 
import subprocess
from agents.mcp import MCPMessage

def generate_llm_response(query, context_chunks):
    prompt = f"Context:\n{'\n'.join(context_chunks)}\n\nQuestion: {query}\nAnswer:"
    result = subprocess.run(["ollama", "run", "llama3:8b-instruct-q4_0"], input=prompt, capture_output=True, text=True)
    return MCPMessage(type="ANSWER", content={"response": result.stdout.strip()})
