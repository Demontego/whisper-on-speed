import logging

import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

from src.core.config import ASRConfig


class ASRChunk(BaseModel):
    text: str
    start_time: float | None = None
    end_time: float | None = None


class ASRonSPEED:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.seconds_per_chunk = config.seconds_per_chunk
        self.device = (
            torch.device(config.device) if config.device != 'cpu' and torch.cuda.is_available() else torch.device('cpu')
        )
        self.is_warmed_up = False
        self.logger = logging.getLogger(__name__)

        self.generate_kwargs = config.generation_config.model_dump()
        self.pipe = self._initialize_model()

    def _initialize_model(self) -> AutomaticSpeechRecognitionPipeline:
        """Initialize the model and pipeline"""
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=self.config.model_id,
            torch_dtype=self.config.get_torch_dtype(),
            low_cpu_mem_usage=self.config.model_settings.low_cpu_mem_usage,
            use_safetensors=self.config.model_settings.use_safetensors,
            attn_implementation=self.config.model_settings.attn_implementation,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.config.model_id)

        return AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.seconds_per_chunk,
            batch_size=self.config.batch_size,
            torch_dtype=self.config.get_torch_dtype(),
            device=self.device,
        )

    def warmup(self, num_warmup_steps: int | None = None) -> None:
        steps = num_warmup_steps or self.config.warmup_steps
        audio = np.ones(self.config.sample_rate * self.seconds_per_chunk) / 2.0
        for _ in range(steps):
            self.pipe(audio, generate_kwargs=self.generate_kwargs, return_timestamps=True)
        self.is_warmed_up = True

    def _process_result(self, result: dict) -> list[ASRChunk]:
        chunks = []
        for chunk in result['chunks']:  # type: ignore
            text = chunk.get('text', '')
            if len(text) == 0:
                continue
            start_time, end_time = chunk.get('timestamp', (None, None))
            if start_time is None and end_time is None and start_time > end_time:
                continue
            chunks.append(
                ASRChunk(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        return chunks

    def process_audio(self, audio: np.ndarray) -> list[ASRChunk]:
        """
        Process one audio file
        Args:
            audio: np.ndarray - audio file
        Returns:
            list[ASRChunk] - list of ASR chunks
        """
        result = self.pipe(audio, generate_kwargs=self.generate_kwargs, return_timestamps=True)
        return self._process_result(result)  # type: ignore

    def process_batch(self, audio: list[np.ndarray]) -> list[list[ASRChunk]]:
        """
        Process a batch of audio files
        Args:
            audio: list[np.ndarray] - list of audio files
        Returns:
            list[list[ASRChunk]] - list of lists of ASR chunks
        """
        results = self.pipe(audio, generate_kwargs=self.generate_kwargs, return_timestamps=True)  # type: ignore
        chunks = []
        for result in results:  # type: ignore
            chunks.append(self._process_result(result))  # type: ignore
        return chunks
