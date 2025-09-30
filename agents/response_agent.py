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

    async def run(self, query: str, stream: bool = False):
        logger.info(f"ResponseAgent running query with stream {stream}: '{query}'")
        try:
            if stream:
                result = ""
                async for chunk in self.chain.astream(query):
                    print(chunk, end="", flush=True)
                    result += chunk
                print()
                logger.info(f"successfully generated a response")
                return result
            else:
                logger.info(f"successfully generated a response")
                return await self.chain.ainvoke(query)    
        except Exception as e:
            logger.error(f"failed on query '{query}': {e}", exc_info=True)
            raise