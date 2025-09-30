import asyncio
from langchain_core.runnables import Runnable

from utils.retriever_factory import make_retriever
from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="retriever_agent", log_file="logs/retriever_agent.log")

class RetrieverAgent(BaseAgent):
    def __init__(self, vector_db, llm):
        super().__init__(
            name="RetrieverAgent", 
            instructions="Retrieve relevant docs"
        )
        self.retriever = make_retriever(vector_db=vector_db, llm=llm)
        logger.info("RetrieverAgent initialized with provided vector db and language model")
    
    async def run(self, query: str, k: int = 5):
        logger.info(f"RetrieverAgent received query: '{query}' with top_k={k}")
        try:
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"retrieved {len(docs)} documents for query '{query}'")

            top_docs = docs[:k]
            for i, doc in enumerate(top_docs, 1):
                snippet = doc.page_content[:100].replace("\n", " ") + "..."
                logger.debug(f"doc {i}: {snippet} | metadata: {doc.metadata}")
            
            return top_docs
        except Exception as e:
            logger.error(f"failed to retrieve documents for query '{query}': {e}")
            return []

class RetrieverRunnable(Runnable):
    def __init__(self, retriever_agent: RetrieverAgent):
        self.retriever_agent = retriever_agent

    def invoke(self, input, config=None):
        # Call the retriever's run method (synchronously)
        docs = asyncio.run(self.retriever_agent.run(input))
        # Convert to string context
        return "\n\n".join([d.page_content for d in docs])
    
    async def ainvoke(self, input, config=None):
        """Optional async version if needed by chain"""
        docs = await self.retriever_agent.run(input)
        return "\n\n".join([d.page_content for d in docs])
