from aqueduct import Flow, FlowStep, BaseTaskHandler, BaseTask

from typing import Optional, TYPE_CHECKING

import numpy as np

from src.core.replica import Replica
from src.core.processor import AudioProcessor


if TYPE_CHECKING:
    from src.core.asr import ASRonSPEED


class AudioTask(BaseTask):
    def __init__(self, audio: np.ndarray):
        super().__init__()
        self.audio: np.ndarray = audio
        self.chunks: Optional[np.ndarray] = None
        self.chunk_starts: Optional[list[float]] = None
        self.tokens: Optional[np.ndarray] = None
        self.replicas: Optional[list[Replica]] = None


class AudioPreProcessorHandler(BaseTaskHandler):
    def __init__(self, processor: AudioProcessor):
        self.processor = processor

    def handle(self, *tasks: AudioTask):
        for task in tasks:
            chunks, seconds = self.processor.preprocess(task.audio)
            task.chunks = chunks
            task.chunk_starts = seconds


class ASRHandler(BaseTaskHandler):
    def __init__(self, asr: "ASRonSPEED", sr: int):
        self.asr = asr
        self.sr = sr

    def handle(self, *tasks: AudioTask):
        all_chunks = []
        chunk_task_map = []

        for i, task in enumerate(tasks):
            all_chunks.append(task.chunks)
            chunk_task_map.extend([i] * len(task.chunk_starts if task.chunk_starts else []))

        chunk_task_map = np.array(chunk_task_map)
        input_batch = np.concatenate(all_chunks, axis=0)
        tokens = self.asr(input_batch, self.sr)

        for i, task in enumerate(tasks):
            task_tokens = tokens[chunk_task_map == i]
            task.tokens = task_tokens


class AudioPostProcessorHandler(BaseTaskHandler):
    def __init__(self, processor: AudioProcessor):
        self.processor = processor

    def handle(self, *tasks: AudioTask):
        for task in tasks:
            replicas = self.processor.postprocess(task.tokens, task.chunk_starts)
            task.replicas = replicas


def get_audio_flow(asr: "ASRonSPEED", processor: AudioProcessor) -> Flow:
    return Flow(
        FlowStep(AudioPreProcessorHandler(processor)),
        FlowStep(ASRHandler(asr, processor.sr), batch_size=20),  # TODO размер батча в конфиг
        FlowStep(AudioPostProcessorHandler(processor)),
        metrics_enabled=False,
    )