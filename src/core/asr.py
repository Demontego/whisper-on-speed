import logging

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from src.core.config import ASRConfig

torch.set_float32_matmul_precision('medium')


class ASRonSPEED:
    def __init__(self, config: ASRConfig) -> None:
        """
        Initialize the model

        Args:
            config: Model configuration
        """
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.dtype = config.get_torch_dtype()
        self.device = torch.device(config.model_settings.device)
        self.is_warmed_up = False
        self.logger = logging.getLogger(__name__)

        self.generate_kwargs = config.generation_config
        self.batch_size = config.batch_size
        self.model = self._initialize_model()

    def _initialize_model(self) -> AutoModelForSpeechSeq2Seq:
        """Initialize the model and pipeline"""
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=self.config.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=self.config.model_settings.low_cpu_mem_usage,
            use_safetensors=self.config.model_settings.use_safetensors,
        ).eval()
        model.to(self.device)
        return model

    def warmup(self, num_warmup_steps: int | None = None, sr: int = 16000) -> None:
        steps = num_warmup_steps or self.config.warmup_steps
        dummy_audio = np.ones((4, sr * 10)) * 0.5
        dummy_tensor = self.processor.feature_extractor(
            dummy_audio, sampling_rate=sr, return_tensors='pt'
        ).input_features
        dummy_tensor = dummy_tensor.to(self.device, dtype=self.dtype)
        for _ in range(steps):
            with sdpa_kernel(SDPBackend.MATH):
                self.model.generate(dummy_tensor, **self.generate_kwargs.model_dump())  # type: ignore
        self.is_warmed_up = True

    def __call__(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Обрабатывает аудио по батчам согласно batch_size из конфига

        Args:
            audio: Входной тензор формы (N, ...)

        Returns:
            list[list[int]]: Результат генерации токенов
        """
        # Получаем общее количество элементов в первой размерности
        input_features = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors='pt').input_features
        total_samples = input_features.shape[0]

        # Если входной батч меньше или равен размеру батча, обрабатываем целиком
        if total_samples <= self.batch_size:
            with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
                batch = input_features.to(self.device, dtype=self.dtype)
                result = self.model.generate(batch, **self.generate_kwargs.model_dump()).cpu().numpy()  # type: ignore
            return result

        # Разбиваем на батчи и обрабатываем по частям
        results = []

        for i in range(0, total_samples, self.batch_size):
            # Извлекаем батч
            end_idx = min(i + self.batch_size, total_samples)
            batch = input_features[i:end_idx]

            # Обрабатываем батч
            with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
                batch = batch.to(self.device, dtype=self.dtype)
                batch_result = self.model.generate(batch, **self.generate_kwargs.model_dump()).cpu()  # type: ignore
                batch_result = F.pad(
                    batch_result,
                    (0, self.generate_kwargs.max_new_tokens - batch_result.shape[1], 0, 0),
                    value=self.config.pad_value,
                )

            results.append(batch_result)

        # Объединяем результаты
        return np.concatenate(results, axis=0)
