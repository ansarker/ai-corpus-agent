import asyncio
from pathlib import Path
from agents.ingestion_agent import IngestionAgent
from agents.embedding_agent import EmbeddingAgent
from agents.retriever_agent import RetrieverAgent, RetrieverRunnable
from agents.response_agent import ResponseAgent
from agents.orchestrator_agent import OrchetratorAgent
from utils.llm_factory import make_llm


async def build_index():
    pdf_dir = Path("test")
    ingestion = IngestionAgent(pdf_dir=pdf_dir)
    embedding = EmbeddingAgent(persist_dir="db_store", model_name="nomic-embed-text")

    documents = await ingestion.run()
    vector_db = await embedding.run(documents=documents, collection_name="corpus_db", overwrite=True)
    return vector_db

async def query_pipeline():
    embedding = EmbeddingAgent(persist_dir="db_store", model_name="nomic-embed-text")
    
    # Check available collections
    await embedding.list_collections()
    
    # Load the vector database
    vector_db = await embedding.load(collection_name="corpus_db")
    llm = make_llm("gemma3", temperature=0.7)

    orchestrator = OrchetratorAgent(vector_db=vector_db, llm=llm)
    
    query = "What are the important takeaways from this book?"
    answer = await orchestrator.run(query=query)
    print("Final answer:\n", answer)

async def chat():
    embedding = EmbeddingAgent(persist_dir="db_store", model_name="nomic-embed-text")
    
    # Check available collections
    await embedding.list_collections()
    
    # Load the vector database
    vector_db = await embedding.load(collection_name="corpus_db")
    llm = make_llm("gemma3", temperature=0.7)

    orchestrator = OrchetratorAgent(vector_db=vector_db, llm=llm)
    
    history = []
    print("\n=== Multi-turn chart started (type 'exit' to quit) ===\n")
    while True:
        user_input = input("❯❯ ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Ending chat. Bye bye")
            break
        history.append({"role": "user", "content": user_input})

        query = user_input
        # [RUN]
        # answer = await orchestrator.run(query=query)
        # print("Assistant:\n", answer)
        # history.append({"role": "assistant", "content": answer})

        # [STREAM]
        async for chunk in orchestrator.stream(query):
            print(chunk["content"], end="", flush=True)


if __name__ == "__main__":
    # asyncio.run(build_index()) # run only once

    # asyncio.run(query_pipeline()) # single query
    
    asyncio.run(chat()) # multi-turn queries