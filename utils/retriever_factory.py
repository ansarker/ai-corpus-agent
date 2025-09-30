from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma

# def make_retriever(vector_db: Chroma, llm, prompt: PromptTemplate) -> MultiQueryRetriever:
#     """Factory to create a retriever with multi-query expansion."""
#     return MultiQueryRetriever.from_llm(
#         retriever=vector_db.as_retriever(),
#         llm=llm,
#         prompt=prompt
#     )

def make_retriever(vector_db: Chroma, llm) -> MultiQueryRetriever:
    """Factory to create a retriever with multi-query expansion."""
    return MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm
    )