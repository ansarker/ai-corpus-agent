from fastapi import APIRouter, HTTPException

from agents.orchestrator_agent import OrchetratorAgent
from utils.logger import get_logger

logger = get_logger(name="ws_server", log_file="logs/ws_server.log")

router = APIRouter()
orchestrator: OrchetratorAgent = None

@router.get("/query")
async def single_query(q: str):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    try:
        response = await orchestrator.run(q)
        return {"query": q, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))