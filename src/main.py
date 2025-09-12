import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request

from src.core.asr import ASRonSPEED
from src.core.config import AqueductConfig, ASRConfig, AudioProcessorConfig, S3Config
from src.core.flow import AudioTask, get_audio_flow
from src.core.processor import AudioProcessor
from src.core.s3_processor import S3Processor

if TYPE_CHECKING:
    import numpy as np
    from aqueduct import Flow

    from src.core.asr import ASRonSPEED
    from src.core.processor import AudioProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict]:
    audio_processor = AudioProcessor(AudioProcessorConfig())

    asr_config = ASRConfig()
    asr = ASRonSPEED(asr_config)

    logger.info('Warmup started...')
    start = time.time()
    asr.warmup(1, audio_processor.sr)
    logger.info(f'Warmup completed in {time.time() - start:.2f} seconds')

    flow = get_audio_flow(asr, audio_processor)
    flow.start()

    s3_processor = S3Processor(S3Config())

    yield {
        'audio_processor': audio_processor,
        'asr': asr,
        'flow': flow,
        's3_processor': s3_processor,
    }


async def audio_processor(request: Request) -> AudioProcessor:
    return request.state.audio_processor


async def asr_model(request: Request) -> ASRonSPEED:
    return request.state.asr


async def aqueduct_flow(request: Request) -> 'Flow':
    return request.state.flow


async def s3_processor(request: Request) -> S3Processor:
    return request.state.s3_processor


async def download_audio_file(s3_processor: S3Processor, file_path: str):
    try:
        audio_np, sr = s3_processor.download_audio(file_path)
        return audio_np, sr
    except Exception as e:
        logger.error(f'Failed to download {file_path}: {e}')
        raise


async def transcribe_audio(flow: 'Flow', audio_np: 'np.ndarray'):
    aq_config = AqueductConfig()
    task = AudioTask(audio_np)
    await flow.process(task, timeout_sec=aq_config.batch_timeout)
    return [{'start_time': seg.start_time, 'end_time': seg.end_time, 'text': seg.text} for seg in task.replicas or []]


async def upload_transcription(s3_processor: S3Processor, segments: list[dict], file_path: str, s3_prefix: str) -> None:
    try:
        output_file = f'{s3_prefix}/{Path(file_path).stem}.json'
        s3_processor.upload_result_json(output_file, json.dumps({'segments': segments}, indent=2))
        logger.info(f'Saved result for {file_path} to {output_file}')
    except Exception as e:
        logger.error(f'Failed to upload {file_path}: {e}')


async def process_audio_file(flow, audio_processor: 'AudioProcessor', s3_processor, file_path: str, s3_prefix: str):
    audio_np, sr = await download_audio_file(s3_processor, file_path)

    if sr != audio_processor.sr:
        logger.warning(f'Skipping {file_path}: sample rate mismatch {sr}')
        return

    segments = await transcribe_audio(flow, audio_np)
    await upload_transcription(s3_processor, segments, file_path, s3_prefix)


async def transcribe_audio_files(
    flow, audio_processor, s3_processor, audio_files: list[str], transcribe_id: str
) -> None:
    s3_prefix = f'results/{transcribe_id}'
    tasks = [process_audio_file(flow, audio_processor, s3_processor, file_path, s3_prefix) for file_path in audio_files]
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.post('/transcribe-s3')
async def transcribe_from_s3(
    path: str,
    flow: 'Flow' = Depends(aqueduct_flow),
    ap: AudioProcessor = Depends(audio_processor),
    s3: S3Processor = Depends(s3_processor),
):
    audio_files = s3.list_input_audio_files(path)
    if not audio_files:
        raise HTTPException(status_code=404, detail='No audio files found')

    transcribe_id = str(uuid.uuid4())
    asyncio.create_task(transcribe_audio_files(flow, ap, s3, audio_files, transcribe_id))

    return {
        'output_path': f'results/{transcribe_id}',
        's3_bucket': s3.output_bucket.name,
        'files_count': len(audio_files),
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=8000)
