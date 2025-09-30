from utils.logger import get_logger

logger = get_logger(name="base_agent", log_file="logs/base_agent.log")

class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
    
    async def run(self, *args, **kwargs):
        """Every agent must implement this method"""
        raise NotImplementedError("Subclasses must implement run()")