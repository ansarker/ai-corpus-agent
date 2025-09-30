import requests
import time
import arxiv
import fitz
import pdfplumber
from requests.exceptions import ChunkedEncodingError, ConnectionError
from pathlib import Path
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import get_logger

# ===== Setup logger =====
logger = get_logger(name="corpus_builder", log_file="logs/corpus_builder.log")

# ===== PDF validator =====
class PDFValidator:
    def __init__(self, directory: Path):
        self.directory = directory
        fitz.TOOLS.mupdf_display_errors(False)
    
    def clean_corrupted_pdfs(self):
        valid_files = []
        for pdf in self.directory.glob("*.pdf"):
            try:
                with fitz.open(pdf) as doc:
                    for page_no in range(len(doc)):
                        _ = doc.load_page(page_no).get_text("text")
                logger.info(f"valid: {pdf.name}")
                valid_files.append(pdf)
            except Exception as e:
                logger.error(f"corrupted: {pdf.name} ({e})")
                try:
                    pdf.unlink()
                    logger.warning(f"deleted: {pdf.name}")
                except Exception as remove_err:
                    logger.error(f"failed not delete {pdf.name}: {remove_err}")
        return valid_files

# ===== PDF extractor =====
class PDFExtractor:
    def __init__(self):
        pass

    @staticmethod
    def extract_pdf(pdf_path: Path) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                texts = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(texts).strip()
            if full_text:
                return full_text
        except Exception:
            logger.warning(f"pdfplumber failed for {pdf_path.name}, falling back to PyMuPDF.")
        
        # fallback to PyMuPDF
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF failed for {pdf_path.name}: {e}")
            return ""

# ===== Text chunker =====
class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 300):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def chunk_text(self, text:str):
        return self.splitter.split_text(text)

# ===== Corpus builder =====
class CorpusBuilder:
    def __init__(self, pdf_dir: Path, output_dir: Path):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.validator = PDFValidator(pdf_dir)
        self.extractor = PDFExtractor()
        self.chunker = TextChunker()
    
    def build_corpus(self):
        valid_pdfs = self.validator.clean_corrupted_pdfs()
        
        for pdf in valid_pdfs:
            text = self.extractor.extract_pdf(pdf)
            if not text:
                logger.warning(f"no text extracted: {pdf.name}")
                continue

            chunks = self.chunker.chunk_text(text)
            
            metadata = {
                "paper_id": pdf.stem,
                "filename": pdf.name,
                "path": str(pdf.resolve())
            }

            corpus_data = [{"text": chunk, "meta": metadata} for chunk in chunks]

            # Save as json
            out_file = self.output_dir/f"{pdf.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"corpus saved: {out_file.name}")

# ===== PDF downloader =====
class PDFDownloader:
    def __init__(self, retries: int = 3, sleep: int = 5, timeout: int = 60):
        self.retries = retries
        self.sleep = sleep
        self.timeout = timeout
    
    def download(self, url: str, dest: Path) -> bool:
        """Download a single PDF with retries and error handling"""
        for attempt in range(1, self.retries + 1):
            try:
                with requests.get(url, stream=True, timeout=self.timeout) as r:
                    if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        return True
                    else:
                        logger.error(f"invalid response {r.status_code} {url}")
                        return False
            except (ChunkedEncodingError, ConnectionError) as e:
                logger.error(f"download error: {e} (attempt {attempt}/{self.retries})")
            
            # Backoff before retry
            if attempt < self.retries:
                time.sleep(self.sleep * attempt)
            else:
                return False
        return False

# ===== Arxiv downloader =====
class ArxivDownloader:
    def __init__(self, output_dir: Path = Path("papers_arxiv"), downloader: PDFDownloader = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader or PDFDownloader()
        self.client = arxiv.Client()

    def download_papers(self, query: str = "geoai", max_results: int =200):
        search = arxiv.Search(
            query=query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in self.client.results(search=search):
            paper_id = result.get_short_id()
            pdf_url = result.pdf_url
            title = result.title
            dest = self.output_dir / f"{paper_id}.pdf"
            
            if dest.exists():
                logger.warning(f"already downloaded: {paper_id}")
                continue
                
            logger.info(f'downloading: {paper_id} {title}')
            success = self.downloader.download(pdf_url, dest)
            if not success:
                logger.warning(f"skipped {paper_id}")
                time.sleep(2)

def main():
    INPUT_DIR = Path("papers_arxiv")
    OUTPUT_DIR = Path("papers_text")

    # Download papers from arxiv
    arxiv_downloader = ArxivDownloader(INPUT_DIR)
    arxiv_downloader.download_papers(query="computer vision", max_results=50)

    builder = CorpusBuilder(pdf_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    builder.build_corpus()

if __name__ == "__main__":
    main()