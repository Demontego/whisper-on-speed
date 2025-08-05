import io
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
import os


import uvicorn
import soundfile as sf
import torch
from cachetools import TTLCache
from fastapi import FastAPI, UploadFile, HTTPException, Request

from src.core.config import ASRConfig, AudioProcessorConfig
from src.core.processor import AudioProcessor
from src.core.asr import ASRonSPEED
from src.core.flow import get_audio_flow, AudioTask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_config = ASRConfig()
    audio_proc_config = AudioProcessorConfig()

    if not torch.cuda.is_available():
        asr_config.model_settings.device = "cpu"
        asr_config.model_settings.torch_dtype = "float32"

    processor = AudioProcessor(audio_proc_config)
    asr = ASRonSPEED(asr_config)
    if os.getenv("SKIP_WARMUP") != "1":   # TODO для отладки
        logger.info("Warmup started...")
        start = time.time()
        asr.warmup(1, processor.sr)
        logger.info(f"Warmup completed in {time.time() - start:.2f} seconds")
    else:
        logger.info("Warmup skipped")

    flow = get_audio_flow(asr, processor)
    flow.start()

    app.state.asr = asr
    app.state.processor = processor
    app.state.flow = flow

    app.state.tasks = TTLCache(maxsize=1000, ttl=3600)  # TODO queue + broker
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(
    request: Request,  # TODO s3 bucket with audiofiles
    file: UploadFile
):
    processor: AudioProcessor = request.app.state.processor
    flow = request.app.state.flow

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/flac"):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio_bytes = await file.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading audio: {e}")

    if sr != processor.sr:
        raise HTTPException(
            status_code=400,
            detail=f"Expected sample rate {processor.sr}, got {sr}"
        )

    task_id = str(uuid.uuid4())

    task = AudioTask(audio_np)  # TODO for file in request.bucket tasks.append(AudioTask(file))


    app.state.tasks[task_id] = {
        "status": "pending",
        "result": None,
        "error": None,
    }

    async def run_task():
        try:
            await flow.process(task, timeout_sec=10.)  # TODO timeout в конфиг
            segments = [
                {"start_time": seg.start_time, "end_time": seg.end_time, "text": seg.text}
                for seg in task.replicas or []
            ]
            app.state.tasks[task_id]["status"] = "completed"
            app.state.tasks[task_id]["result"] = {"segments": segments}
        except Exception as e:
            app.state.tasks[task_id]["status"] = "failed"
            app.state.tasks[task_id]["error"] = str(e)
    
    asyncio.create_task(run_task())

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(404, detail="Task ID not found")

    return {
        "status": task["status"],
        "result": task["result"],
        "error": task["error"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
