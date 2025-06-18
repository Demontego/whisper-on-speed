import logging

import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline


class ASRChunk(BaseModel):
    text: str
    start_time: float | None = None
    end_time: float | None = None


class ASRonSPEED:
    def __init__(self, model_id: str, seconds_per_chunk: int = 10, batch_size: int = 16):
        self.seconds_per_chunk = seconds_per_chunk
        self.device = torch.device('cuda')
        self.is_warmed_up = False
        self.logger = logging.getLogger(__name__)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation='sdpa',
        )
        model.to(self.device)

        self.generate_kwargs = {
            'max_new_tokens': 324,
            'num_beams': 3,
            'condition_on_prev_tokens': True,
            'compression_ratio_threshold': 1.35,
            'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            'logprob_threshold': -1.0,
            'use_cache': True,
        }

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.seconds_per_chunk,
            batch_size=batch_size,
            torch_dtype=torch.float16,
            device=self.device,
        )

    def warmup(self, num_warmup_steps: int = 3) -> None:
        """
        Warmup the model
        Args:
            num_warmup_steps: int - number of warmup steps
        """
        audio = np.ones(16_000 * self.seconds_per_chunk) / 2.0
        for _ in range(num_warmup_steps):
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

    def procces_audio(self, audio: np.ndarray) -> list[ASRChunk]:
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
