from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="summarizer_agent", log_file="logs/summarizer_agent.log")

class SummarizerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            name="SummarizerAgent",
            instructions="Summarizes research papers concisely"
        )
        self.llm = llm

    async def run(self, document, stream: bool = False, max_chars: int = 2000):
        logger.info(f"starting summarization for document")

        prompt = f"""
        You are a research assistant. Summarize the following paper concisely, highlighting key insights:

        Document (up to {max_chars} characters):
        {document.page_content[:max_chars]}

        Provide the summary in clear, concise language.
        """

        try:
            if stream:
                logger.info("streaming summary...")
                summary = ""
                async for chunk in self.llm.astream(prompt):
                    print(chunk.content, end="", flush=True)
                    summary += chunk.content
                print()
            else:
                logger.info("invoking LLM for summarization...")
                summary = await self.llm.ainvoke(prompt)

            logger.info("summarization completed successfully")
            return summary

        except Exception as e:
            logger.error(f"summarization failed: {e}")
            return "error: summarization failed"
