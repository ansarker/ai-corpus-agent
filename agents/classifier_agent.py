from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="classifier_agent", log_file="logs/classifier_agent.log")

class ClassifierAgent(BaseAgent):
    def __init__(self, llm):
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

    async def run(self, document, stream: bool = False):
        logger.info(f"starting classification for document with metadata: {document.metadata}")

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
            if stream:
                logger.info(f"streaming classification response...")
                classification = ""
                async for chunk in self.llm.astream(prompt):
                    print(chunk.content, end="", flush=True)
                    classification += chunk.content
                print()
            else:
                logger.info("invoking LLM for classification...")
                classification = await self.llm.ainvoke(prompt)

            logger.info(f"classification done")
            return classification
        except Exception as e:
            logger.error(f"classification failed: {e}")
            return {"discipline": "Other"}