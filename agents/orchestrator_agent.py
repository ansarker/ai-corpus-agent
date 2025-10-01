from typing import AsyncGenerator, Union, Dict, List, Any

from .base_agent import BaseAgent
from .retriever_agent import RetrieverAgent
from .response_agent import ResponseAgent
from .summarizer_agent import SummarizerAgent
from .classifier_agent import ClassifierAgent
from utils.logger import get_logger


logger = get_logger(name="orchestrator_agent", log_file="logs/orchestrator_agent.log")

class OrchetratorAgent(BaseAgent):
    def __init__(self, vector_db, llm):
        super().__init__(
            name="OrchestratorAgent", 
            instructions="Directs queries to the right agents"
        )
        self.retriever_agent = RetrieverAgent(vector_db, llm)
        self.rag_agent = ResponseAgent(llm, self.retriever_agent.retriever)
        self.summarizer_agent = SummarizerAgent(llm)
        self.classifier_agent = ClassifierAgent(llm)
        
        logger.info("OrchestratorAgent initialized with Retriever, RAG, Summarizer, and Classifier agents.")

    async def run(self, query: str):
        logger.info(f"[RUN] received query: '{query}'")
        try:
            agent, doc_needed = self._route(query=query)
            if doc_needed:
                docs = await self.retriever_agent.run(query=query)
                if not docs:
                    logger.warning(f"[RUN] no documents retrieved for {agent.__class__.__name__}")
                    return f"no documents available to {agent.name.lower()}"
                result = await agent.run(docs[0])
            else:
                result = await agent.run(query)
                
            logger.info(f"[RUN] {agent.name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"[RUN] error while processing query '{query}': {e}", exc_info=True)
            return f"an error occurred while handling the query: {e}"
    
    async def stream(self, query: str) -> AsyncGenerator[Union[str, Dict[str, str], List[Any]], None]:
        logger.info(f"[STREAM] received query: '{query}'")
        try:
            agent, doc_needed = self._route(query=query)
            if doc_needed:
                docs = await self.retriever_agent.run(query=query)
                if not docs:
                    logger.warning(f"[STREAM] no documents retrieved for {agent.__class__.__name__}")
                    yield f"no documents available to {agent.name.lower()}"
                    return
                async for chunk in agent.stream(docs[0]):
                    yield chunk
            else:
                async for chunk in agent.stream(query):
                    yield chunk
            logger.info(f"[STREAM] {agent.name} completed successfully")
        except Exception as e:
            logger.error(f"[STREAM]] error processing query '{query}': {e}", exc_info=True)
            yield f"an error occurred handling the query: {e}"

    def _route(self, query: str):
        """
        Determines which agent to use based on the query.
        Returns a tuple: (agent_instance, needs_document: bool)
        """
        query_lower = query.lower()
        if "summarize" in query_lower:
            logger.debug("[ROUTE] Routing to SummarizerAgent")
            return self.summarizer_agent, True
        elif "classify" in query_lower:
            logger.debug("[ROUTE] Routing to ClassifierAgent")
            return self.classifier_agent, True
        else:
            logger.debug("[ROUTE] Routing to ResponseAgent (RAG)")
            return self.rag_agent, False