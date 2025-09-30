# Multi-Agent RAG Pipeline

This repository implements a modular, multi-agent architecture for document ingestion, embedding, retrieval, and reasoning.

It is built around LangChain and Ollama model, and allows multi-turn, query-driven orchestration with RAG (retrieval-augmented generation), summarization, and classification capabilities.

### Features

* **IngestionAgent** → Loads raw documents (e.g. PDFs).
* **EmbeddingAgent** → Embeds and stores documents in a vector DB (db_store/).
* **RetrieverAgent** → Fetches top-k relevant documents for a query.
* **ResponseAgent (RAG)** → Generates answers with context from the retriever.
* **SummarizerAgent** → Summarizes retrieved content.
* **ClassifierAgent** → Classifies content into categories.
* **OrchestratorAgent** → Routes queries to the right agent depending on intent.

### Installation

1. **Clone the repo**
    ```bash
    git clone https://github.com/yourname/ai-corpus-agent.git
    cd ai-corpus-agent
    ```
2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install dependencies**

    I forgot what I installted. Do yourself by getting error.
    ```bash
    pip install -r requirements.txt
    ```
    Note: `requirements.txt` is empty 😑