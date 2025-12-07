from pydantic import BaseModel
from typing import Optional

class UserRequest(BaseModel):
    message: str
    max_tokens: int = 200
    temperature: float = 0.5
    system_instruction: Optional[str] = None
    language: str = "en"
class BotResponse(BaseModel):
    response: str
class IndexSwitchRequest(BaseModel):
    index_name: str