from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import shutil
import chromadb

from corpus_loader import CorpusLoader
from utils.logger import get_logger

logger = get_logger(name="vectorstore_builder", log_file="logs/vectorstore_builder.log")

class VectorStoreBuilder:
    def __init__(self, persist_dir: Path, model_name: str = "nomic-embed-text"):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.model_name)
    
    def build_vectorstore(self, documents: list[Document], collection_name: str = "corpus_vector_db", overwrite: bool = False) -> Chroma:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            logger.info(f"overwriting existing vector store")
            shutil.rmtree(self.persist_dir, ignore_errors=True)
        
        vector_db = Chroma.from_documents(
            collection_name=collection_name,
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

        logger.info(f"vector store built with {len(documents)} documents")
        return vector_db
    
    def load_vectorstore(self, collection_name: str = "langchain") -> Chroma:
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        logger.info(f"loaded vector store with {vector_db._collection.count()} data from {self.persist_dir}")
        return vector_db

    def list_collections(self):
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collections = client.list_collections()
        logger.info(f"available collections: {[collection.name for collection in collections]}")

def main():
    corpus_dir = Path("papers_text")
    corpus_loader = CorpusLoader(corpus_dir=corpus_dir)
    documents = corpus_loader.load_documents()

    vectorstore_builder = VectorStoreBuilder(persist_dir=Path("noidea"))
    vector_db = vectorstore_builder.build_vectorstore(documents=documents, collection_name="corpus_db", overwrite=True)

    print(vector_db)

if __name__ == "__main__":
    main()