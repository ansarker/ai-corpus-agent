from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from agents.embedding_agent import EmbeddingAgent
from agents.orchestrator_agent import OrchetratorAgent
from utils.llm_factory import make_llm
from utils.logger import get_logger
from .routes import chat, query
from .config import settings


logger = get_logger(name="ws_server", log_file="logs/ws_server.log")

vector_db: Optional[Chroma] = None
orchestrator: Optional[OrchetratorAgent] = None
llm: Optional[ChatOllama] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, orchestrator, llm
    try:
        llm = make_llm(settings.MODEL_NAME, temperature=0.7)
        embedding = EmbeddingAgent(
            persist_dir=settings.PERSIST_DIR, 
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        vector_db = await embedding.load(collection_name=settings.COLLECTION_NAME)
        orchestrator = OrchetratorAgent(vector_db=vector_db, llm=llm)

        query.orchestrator = orchestrator
        chat.orchestrator = orchestrator

        yield
    except Exception as e:
        logger.error(f"[LIFESPAN] failed to initialize orchestrator: {e}")
        yield

app = FastAPI(title="AI Corpus ML Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(query.router)
app.include_router(chat.router)

# @app.websocket("/ws")
# async def ws_endpoint(ws: WebSocket):
#     await ws.accept()
#     session_id = str(uuid.uuid4())
#     logger.info(f"[SERVER] New WebSocket session {session_id} connected")

#     try:
#         while True:
#             msg = await ws.receive_text()
#             data = json.loads(msg)
#             query = data.get("content", "")
        
#             logger.info(f"[SERVER] [{session_id}] received query '{query}'")

#             try:
#                 assert orchestrator is not None
#                 async for chunk in orchestrator.stream(query):
#                     content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
#                     await ws.send_text(json.dumps({
#                         "type": "chunk", 
#                         "content": content
#                     }))
#                 await ws.send_text(json.dumps({"type": "done"}))
#             except Exception as e:
#                 await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
#     except Exception as e:
#         logger.error(f"[SERVER] Unexpected error in session {session_id}: {e}")
#     finally:
#         logger.info(f"[SERVER] [{session_id}] connection closed")

@app.get("/")
async def health():
    return {"message": "Websocker server alive."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(host=f"{settings.HOST}", app="api.server:app", port=settings.PORT, reload=True, workers=4)