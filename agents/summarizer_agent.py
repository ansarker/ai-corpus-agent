from langchain_ollama import ChatOllama

from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="summarizer_agent", log_file="logs/summarizer_agent.log")

class SummarizerAgent(BaseAgent):
    def __init__(self, llm: ChatOllama):
        super().__init__(
            name="SummarizerAgent",
            instructions="Summarizes research papers concisely"
        )
        self.llm = llm

    async def run(self, document, max_chars: int = 2000):
        logger.info(f"[RUN] starting summarization for document")
        prompt = f"""
        You are a research assistant. Summarize the following paper concisely, highlighting key insights:

        Document (up to {max_chars} characters):
        {document.page_content[:max_chars]}

        Provide the summary in clear, concise language.
        """
        try:
            summary = await self.llm.ainvoke(prompt)
            logger.info("[RUN] summarization completed successfully")
            return {"type": "summary", "content": summary.content, "metadata": summary.response_metadata}
        except Exception as e:
            logger.error(f"[RUN] summarization failed: {e}")
            return {"type": "summary", "content": "error: summarization failed", "metadata": {}}

    async def stream(self, document, max_chars: int = 2000):
        logger.info(f"[STREAM] starting summarization for document")
        prompt = f"""
        You are a research assistant. Summarize the following paper concisely, highlighting key insights:

        Document (up to {max_chars} characters):
        {document.page_content[:max_chars]}

        Provide the summary in clear, concise language.
        """
        try:
            async for chunk in self.llm.astream(prompt):
                yield {"type": "summary", "content": chunk.content, "metadata": chunk.response_metadata}
            logger.info("[STREAM] summarization completed successfully")
        except Exception as e:
            logger.error(f"[STREAM] summarization failed: {e}")
            yield {"type": "summary", "content": "error: summarization failed", "metadata": {}}
