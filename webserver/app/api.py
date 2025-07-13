from fastapi import FastAPI, HTTPException
from .tasks import process_audio
from .schemas import AddTaskRequest
from .config import settings
from .celery_app import celery, check_queue_empty, shutdown_gcp_instance

app = FastAPI()

@app.post("/add_task")
async def add_task(request: AddTaskRequest):
    if request.bucket_name != settings.allowed_bucket:
        raise HTTPException(403, detail="Bucket not allowed")
    
    task = process_audio.delay(request.bucket_name, request.path)
    return {"task_id": task.id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    result = celery.AsyncResult(task_id)
    return {
        "status": result.state,
        "result": result.result if result.ready() else None,
        "error": str(result.traceback) if result.failed() else None,
    }

@app.get("/shutdown")
async def check_shutdown():
    if check_queue_empty():
        shutdown_gcp_instance()
        return {"status": "shutting down"}
    return {"status": "busy"}