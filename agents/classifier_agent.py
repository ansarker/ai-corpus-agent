
from .base_agent import BaseAgent

class ClassifierAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            name="ClassifierAgent", 
            instructions="Classifies papers into disciplines"
        )
        self.llm = llm

    async def run(self, document):
        prompt = f"Classify this paper into disciplines (AI, CV, NLP, etc.):\n\n{document.page_content}"
        return await self.llm.ainvoke(prompt)
