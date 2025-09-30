from langchain_ollama import ChatOllama

def make_llm(model: str = "gemma3", temperature: float = 0.3) -> ChatOllama:
    """Factory to create an Ollama LLM instance."""
    return ChatOllama(model=model, temperature=temperature)
