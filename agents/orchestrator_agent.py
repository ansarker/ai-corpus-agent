from .base_agent import BaseAgent
from .retriever_agent import RetrieverAgent
from .response_agent import ResponseAgent
from .summerizer_agent import SummarizerAgent
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
        logger.info(f"received query: '{query}'")
        try:
            if "summarize" in query.lower():
                logger.debug("routing query to SummarizerAgent")
                docs = await self.retriever_agent.run(query)
                if not docs:
                    logger.warning(f"no documents retrieved for summarization.")
                    return "no documents available to summarize"
                result = await self.summarizer_agent.run(docs[0])
                logger.info("SummarizerAgent completed successfully")
                return result
            elif "classifier" in query.lower():
                logger.debug("routing query to ClassifierAgent")
                docs = await self.retriever_agent.run(query)
                if not docs:
                    logger.warning(f"no documents retrieved for classification.")
                    return "no documents available to classify"
                result = await self.classifier_agent.run(docs[0])
                logger.info("ClassifierAgent completed successfully")
                return result
            else:
                logger.debug("routing query to ResponseAgent (RAG)")
                result = await self.rag_agent.run(query, stream=True)
                logger.info("ResponseAgent (RAG) completed successfully")
                return result
        except Exception as e:
            logger.error(f"error while processing query '{query}': {e}", exc_info=True)
            return f"an error occurred while handling the query: {e}"
