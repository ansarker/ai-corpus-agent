from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json, uuid

from agents.orchestrator_agent import OrchetratorAgent
from utils.logger import get_logger

logger = get_logger(name="ws_server", log_file="logs/ws_server.log")

router = APIRouter()
orchestrator: OrchetratorAgent = None

@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS SERVER] New WebSocket session {session_id} connected")

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            query = data.get("content", "")
        
            logger.info(f"[WS SERVER] [{session_id}] received query '{query}'")

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
        logger.error(f"[WS SERVER] Unexpected error in session {session_id}: {e}")
    finally:
        logger.info(f"[WS SERVER] [{session_id}] connection closed")
