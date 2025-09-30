from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from utils.logger import get_logger
from .base_agent import BaseAgent


logger = get_logger(name="ingestion_agent", log_file="logs/ingestion_agent.log")

class IngestionAgent(BaseAgent):
    def __init__(self, pdf_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(
            name="IngestionAgent",
            instructions="Load and preprocess PDF documents into chunks"
        )
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    async def run(self) -> List[Document]:
        documents: List[Document] = []
        logger.info(f"starting ingestion from {self.pdf_dir}")
        
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()
                docs = self.splitter.split_documents(pages)
                for d in docs:
                    d.metadata["source"] = pdf_file.name
                documents.extend(docs)
                logger.info(f"loaded {len(docs)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"failed to load {pdf_file.name}: {e}")
        
        logger.info(f"ingestion complete: {len(documents)} total chunks")
        return documents