from pydantic import BaseModel


class Replica(BaseModel):
    text: str
    start_time: float | None = None
    end_time: float | None = None
