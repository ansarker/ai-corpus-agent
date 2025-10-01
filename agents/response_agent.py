from typing import AsyncGenerator
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="response_agent", log_file="logs/response_agent.log")

class ResponseAgent(BaseAgent):
    def __init__(self, llm, retriever):
        super().__init__(
            name="ResponseAgent",
            instructions="Answer questions with retrieved docs"
        )
        self.llm = llm
        self.retriever = retriever
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful research assistant. Use the following context to answer:
            
            Context:
            {context}

            Question:
            {question}

            Answer clearly and concisely."""
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("ResponseAgent initialized with provided LLM and retriever")

    async def run(self, query: str):
        logger.info(f"[RUN] received query: '{query}'")
        try:
            result = await self.chain.ainvoke(query)
            logger.info(f"[RUN] successfully generated a response")
            return {
                "type": "response",
                "content": result,
                "metadata": {"model": getattr(self.chain, "llm_name", "unknown")}
            }
        except Exception as e:
            logger.error(f"[RUN] failed on query '{query}': {e}", exc_info=True)
            return {"type": "response", "content": "error: failed to respond", "metadata": {}}
    
    async def stream(self, query: str) -> AsyncGenerator[dict, None]:
        logger.info(f"[STREAM] received query: '{query}'")
        try:
            async for chunk in self.chain.astream(query):
                yield {"type": "response", "content": chunk, "metadata": {}}
            logger.info(f"[STREAM] successfully generated a response")
        except Exception as e:
            logger.error(f"[STREAM] failed on query '{query}': {e}", exc_info=True)
            yield {"type": "response", "content": "error: failed to respond", "metadata": {}}
