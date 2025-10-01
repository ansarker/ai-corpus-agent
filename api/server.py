from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from contextlib import asynccontextmanager
import json, uuid
import asyncio
from typing import Optional
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


from agents.embedding_agent import EmbeddingAgent
from agents.orchestrator_agent import OrchetratorAgent
from utils.llm_factory import make_llm
from utils.logger import get_logger
from . import config


logger = get_logger(name="ws_server", log_file="logs/ws_server.log")

vector_db: Optional[Chroma] = None
orchestrator: Optional[OrchetratorAgent] = None
llm: Optional[ChatOllama] = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global vector_db, orchestrator, llm
    llm = make_llm(config.MODEL_NAME, temperature=0.3)
    embedding = EmbeddingAgent(
        persist_dir=config.PERSIST_DIR, 
        model_name=config.EMBEDDING_MODEL_NAME
    )
    vector_db = await embedding.load(collection_name=config.COLLECTION_NAME)
    orchestrator = OrchetratorAgent(vector_db=vector_db, llm=llm)
    yield

app = FastAPI(title="AI Corpus ML Service", lifespan=lifespan)
origins = [
    "http://localhost.com",
    "http://127.0.0.1",
    "http://localhost.com:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[SERVER] New WebSocket session {session_id} connected")

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            query = data.get("content", "")
        
            logger.info(f"[SERVER] [{session_id}] received query '{query}'")

            try:
                assert orchestrator is not None
                async for chunk in orchestrator.stream(query):
                    content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
                    await ws.send_text(json.dumps({
                        "type": "chunk", 
                        "content": content
                    }))
                await ws.send_text(json.dumps({"type": "done"}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
    except Exception as e:
        logger.error(f"[SERVER] Unexpected error in session {session_id}: {e}")
    finally:
        logger.info(f"[SERVER] [{session_id}] connection closed")

@app.get("/")
async def health():
    return {"message": "Websocker server alive."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(host="0.0.0.0", app="api.server:app", port=8080, reload=True, workers=4)