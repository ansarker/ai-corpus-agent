import argparse
import asyncio
from pathlib import Path
from agents.ingestion_agent import IngestionAgent
from agents.embedding_agent import EmbeddingAgent
from agents.orchestrator_agent import OrchetratorAgent
from utils.llm_factory import make_llm


async def build_index(args):
    pdf_dir = Path(args.path)
    ingestion = IngestionAgent(pdf_dir=pdf_dir)
    embedding = EmbeddingAgent(persist_dir="db_store", model_name="nomic-embed-text")

    documents = await ingestion.run()
    vector_db = await embedding.run(documents=documents, collection_name="corpus_db", overwrite=True)
    return vector_db

async def query_pipeline(args):
    embedding = EmbeddingAgent(persist_dir="db_store", model_name="nomic-embed-text")
    
    # Check available collections
    await embedding.list_collections()
    
    # Load the vector database
    vector_db = await embedding.load(collection_name="corpus_db")
    llm = make_llm("gemma3", temperature=0.7)

    orchestrator = OrchetratorAgent(vector_db=vector_db, llm=llm)
    answer = await orchestrator.run(query=args.query)
    print("Response:\n" + answer["content"].strip())

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

        # [STREAM]
        async for chunk in orchestrator.stream(query):
            print(chunk["content"], end="", flush=True)

def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Corpus Agent Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === Build command ===
    build_parser = subparsers.add_parser('build', help="Build the vector database")
    build_parser.add_argument('-p', '--path', required=True, help="Directory path of books/papers etc")
    
    # === Query command ===
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("-q", "--query", required=True, help="Query string to run")

    # === Chat command ===
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")

    args = parser.parse_args()
    return args

def main():
    args = init_parser()
    
    # === Dispatch ===
    if args.command == "build":
        asyncio.run(build_index(args))
    elif args.command == "query":
        asyncio.run(query_pipeline(args))
    elif args.command == "chat":
        asyncio.run(chat())
    else:
        args.print_help()

if __name__ == "__main__":
    main()