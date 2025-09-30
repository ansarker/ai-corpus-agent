from .base_agent import BaseAgent

class SummarizerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            name="SummarizerAgent",
            instructions="Summarizes research papers"
        )
        self.llm = llm

    async def run(self, document):
        prompt = f"Summarize the following paper:\n\n{document.page_content}"
        return await self.llm.ainvoke(prompt)
