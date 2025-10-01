from typing import AsyncGenerator
from langchain_ollama import ChatOllama
from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="classifier_agent", log_file="logs/classifier_agent.log")

class ClassifierAgent(BaseAgent):
    def __init__(self, llm: ChatOllama):
        super().__init__(
            name="ClassifierAgent", 
            instructions="Classifies papers into disciplines"
        )
        self.llm = llm
        self.allowed_labels = [
            # STEM
            "Artificial Intelligence",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics",
            "Machine Learning",
            "Data Science",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Medicine",
            "Engineering",
            "Environmental Science",

            # Social Sciences
            "Economics",
            "Psychology",
            "Sociology",
            "Political Science",
            "Education",

            # Humanities
            "Philosophy",
            "History",
            "Linguistics",
            "Literature",
            "Arts",

            # Fallback
            "Other"
        ]

    async def run(self, document) -> dict:
        logger.info(f"[RUN] starting classification for document with metadata: {document.metadata}")
        prompt = f"""
        You are a research paper classifier.
        Task:
        - Assign the MOST relevant discipline(s) from the list below.
        - If multiple apply, choose the most specific one.
        - If unsure, use "Other".
        
        Allowed categories:
        {", ".join(self.allowed_labels)}

        Output strictly in JSON format:
        {{"discipline": "chosen category"}}

        Document (first 1000 chars shown):
        {document.page_content[:1000]}
        """
        try:
            classification = await self.llm.ainvoke(prompt)
            logger.info(f"[RUN] classification done")
            return {"type": "classification", "content": classification.content, "metadata": classification.response_metadata}
        except Exception as e:
            logger.error(f"[RUN] classification failed: {e}")
            return {"type": "classification", "content": "Other", "metadata": {"allowed_labels": self.allowed_labels}}

    async def stream(self, document) -> AsyncGenerator[dict, None]:
        logger.info(f"[STREAM] starting classification for document with metadata: {document.metadata}")
        prompt = f"""
        You are a research paper classifier.
        Task:
        - Assign the MOST relevant discipline(s) from the list below.
        - If multiple apply, choose the most specific one.
        - If unsure, use "Other".
        
        Allowed categories:
        {", ".join(self.allowed_labels)}

        Output strictly in JSON format:
        {{"discipline": "chosen category"}}

        Document (first 1000 chars shown):
        {document.page_content[:1000]}
        """
        try:
            async for chunk in self.llm.astream(prompt):
                yield {"type": "classification", "content": chunk.content, "metadata": {"allowed_labels": self.allowed_labels}}
            logger.info(f"[STREAM] classification done")
        except Exception as e:
            logger.error(f"[STREAM] classification failed: {e}")
            yield {"type": "classification", "content": "Other", "metadata": {"allowed_labels": self.allowed_labels}}