import asyncio
import websockets
import json
from utils.logger import get_logger

logger = get_logger(name="test_websocket", log_file="logs/test_websocker.log")

async def test_ws():
    uri = "ws://localhost:8080/ws"
    query_payload = {
        "id": "q1",
        "type": "query",
        # "content": "Summarize this book",
        "content": "What are the most important takeaways from this book?",
        # "content": "can you classify the areas of this book?"
    }
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(query_payload))
        logger.info(f"[TEST WS] sent query: {query_payload['content']}")

        final_response = ""
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"[TEST WS] received non-JSON message: {message}")
                continue

            if data.get("type") == "chunk":
                print(data["content"], end="", flush=True)
            elif data.get("type") == "done":
                print("\nStream complete")
                break
        print("\n--- Final response collected ---")
        logger.info(f"Final response: {final_response}")

if __name__ == "__main__":
    asyncio.run(test_ws())
