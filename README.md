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
    git clone https://github.com/ansarker/ai-corpus-agent.git
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


### Usage

#### CLI

All commands are run from the project root with:
```bash
python main.py <command> [options]
```

#### Build vector database
Builds a persistent vector database from a directory of documents (books, papers, etc.).
```bash
python main.py build -p ./data/books
```
**Options:**

* `-p`, `--path` (**required**) → Path to directory containing `.pdf`, `.txt`, `.md`, or other supported files.

#### Single Query Mode

Run a one-off query against the knowledge base.
```bash
python main.py query -q "What are the key takeaways from Atomic Habits?"
```

**Options:**

`-q`, `--query` (**required**) → The question or query to run.

#### Interactive Chat Mode
Start a conversational session with the AI (multi-turn dialogue).
```bash
python main.py chat
```
**Behavior:**
* Starts a streaming chat loop.
* Type your questions one by one.
* Use `Ctrl+C` or type `exit`, `quit` to exit.

#### Example
```bash
# Build index from a research papers directory
python main.py build -p ./corpus/papers

# Ask a single query
python main.py query -q "Summarize the methods section of the corpus"

# Start an interactive chat session
python main.py chat
```