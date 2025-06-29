import logging

import numpy as np
import torch
from transformers import AutoProcessor

from src.core.config import AudioProcessorConfig
from src.core.replica import Replica


class AudioProcessor:
    def __init__(self, config: AudioProcessorConfig) -> None:
        """
        Initialize the audio processor

        Args:
            config: Audio processor configuration
        """
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.vad, utils = torch.hub.load(
            repo_or_dir=config.model_vad, model='silero_vad', force_reload=False, onnx=True
        )
        self.get_speech_timestamps = utils[0]
        self.sr = config.sr
        self.chunk_sec = config.chunk_sec
        self.overlap_sec = config.overlap_sec
        self.logger = logging.getLogger(__name__)

    def preprocess(self, audio: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """
        Cuts the audio array into chunks with overlap and returns an array of chunks.

        :param audio: 1D np.ndarray of audio signal (float32 or int16)
        :return: np.ndarray of shape (N_chunks, chunk_len) with dtype float32
        """
        if audio.ndim != 1:
            self.logger.error('Input audio must be a 1D np.ndarray.')
            raise ValueError('Input audio must be a 1D np.ndarray.')

        chunk_len = int(self.sr * self.chunk_sec)
        step = int(chunk_len - self.sr * self.overlap_sec)
        total_len = len(audio)

        chunks = []
        seconds = []

        for start in range(0, total_len, step):
            end = start + chunk_len
            chunk = audio[start:end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            speech_timestamps = self.get_speech_timestamps(chunk, self.vad, return_seconds=True)
            sum_talk_seconds = sum(item['end'] - item['start'] for item in speech_timestamps)
            if speech_timestamps and sum_talk_seconds > 1.0:
                chunks.append(chunk)
                seconds.append(start / self.sr)

        return np.stack(chunks), seconds

    def postprocess(self, tokens: np.ndarray, seconds: list[float]) -> list[Replica]:
        segments = []
        for token, second in zip(tokens, seconds):
            _, chunks = self.processor.tokenizer._decode_asr(
                [{'tokens': [token]}], return_language=False, return_timestamps=True, time_precision=0.02
            )
            for chunk in chunks['chunks']:
                segments.append(
                    Replica(
                        start_time=chunk['timestamp'][0] or 0 + second,
                        end_time=min(chunk['timestamp'][1] or self.chunk_sec, self.chunk_sec) + second,
                        text=chunk['text'].strip(),
                    )
                )

        return segments
