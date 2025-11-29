from pydantic import BaseModel

class UserRequest(BaseModel):
    message: str
    max_tokens: int = 200
    temperature: float = 0.1

class BotResponse(BaseModel):
    response: str
