from pydantic import BaseModel
from typing import Optional

class AddTaskRequest(BaseModel):
    bucket_name: str
    path: str

class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[dict]
    error: Optional[str]