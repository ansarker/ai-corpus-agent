from pathlib import Path
from vector_store import VectorStoreBuilder

from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="embedding_agent", log_file="logs/embedding_agent.log")

class EmbeddingAgent(BaseAgent):
    def __init__(self, persist_dir: str = "db_store", model_name: str = "nomic-embed-text"):
        super().__init__(
            name="EmbeddingAgent", 
            instructions="Embeds docs into vector database"
        )
        self.builder = VectorStoreBuilder(persist_dir=Path(persist_dir), model_name=model_name)
    
    async def run(self, documents: list, collection_name: str = "corpus_db", overwrite: bool = False):
        logger.info(f"starting embedding process for collection='{collection_name}' "
                    f"with {len(documents)} documents. Overwrite={overwrite}")
        try:
            vector_store = self.builder.build_vectorstore(documents, collection_name, overwrite)
            logger.info(f"embedding complete: stored {len(documents)} documents into '{collection_name}'")
            return vector_store
        except Exception as e:
            logger.error(f"embedding failed for collection='{collection_name}': {e}")
            raise
    
    async def load(self, collection_name: str = "corpus_db"):
        return self.builder.load_vectorstore(collection_name)
    
    async def list_collections(self):
        return self.builder.list_collections()