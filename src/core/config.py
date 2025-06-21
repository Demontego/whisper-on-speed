from typing import Tuple

import torch
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Конфигурация для загрузки модели"""

    torch_dtype: str = 'float16'
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    attn_implementation: str = 'sdpa'


class GenerationConfig(BaseModel):
    """Конфигурация для генерации текста"""

    max_new_tokens: int = Field(default=324, description='Максимальное количество новых токенов')
    num_beams: int = Field(default=3, description='Количество лучей для beam search')
    condition_on_prev_tokens: bool = Field(default=True, description='Учитывать предыдущие токены')
    compression_ratio_threshold: float = Field(default=1.35, description='Порог сжатия')
    temperature: Tuple[float, ...] = Field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), description='Температуры для генерации'
    )
    logprob_threshold: float = Field(default=-1.0, description='Порог логарифмической вероятности')
    use_cache: bool = Field(default=True, description='Использовать кэш')


class ASRConfig(BaseModel):
    """Основная конфигурация для ASR"""

    model_id: str = Field(default='openai/whisper-large-v3-turbo', description='ID модели для загрузки')
    seconds_per_chunk: int = Field(default=10, description='Количество секунд на чанк')
    batch_size: int = Field(default=16, description='Размер батча')
    device: str = Field(default='cuda', description='Устройство для вычислений')
    warmup_steps: int = Field(default=3, description='Количество шагов прогрева')
    sample_rate: int = Field(default=16_000, description='Частота дискретизации аудио')

    # Renamed from model_config to avoid conflict with Pydantic's reserved attribute
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    def get_torch_dtype(self):
        """Получить torch dtype из строкового представления"""
        # Для CPU принудительно используем float32, так как float16 неэффективен
        if self.device == 'cpu' or not torch.cuda.is_available():
            return torch.float32

        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
        }
        return dtype_map.get(self.model_settings.torch_dtype, torch.float16)
