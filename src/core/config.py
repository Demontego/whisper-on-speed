from typing import Tuple

import torch
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model loading"""

    torch_dtype: str = 'float16'
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True


class GenerationConfig(BaseModel):
    """Configuration for text generation"""

    max_new_tokens: int = Field(default=324, description='maximum number of new tokens')
    num_beams: int = Field(default=3, description='number of beams for beam search')
    condition_on_prev_tokens: bool = Field(default=False, description='consider previous tokens')
    compression_ratio_threshold: float = Field(default=1.35, description='compression ratio threshold')
    temperature: Tuple[float, ...] = Field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), description='temperatures for generation'
    )
    logprob_threshold: float = Field(default=-1.0, description='log probability threshold')


class ASRConfig(BaseModel):
    """Основная конфигурация для ASR"""

    model_id: str = Field(default='openai/whisper-large-v3-turbo', description='model id')
    seconds_per_chunk: int = Field(default=10, description='seconds per chunk')
    batch_size: int = Field(default=16, description='batch size')
    device: str = Field(default='cuda', description='device')
    warmup_steps: int = Field(default=3, description='number of warmup steps')
    sample_rate: int = Field(default=16_000, description='sample rate')

    # Renamed from model_config to avoid conflict with Pydantic's reserved attribute
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    def get_torch_dtype(self):
        """Get torch dtype from string representation"""
        # For CPU, force float32, as float16 is inefficient
        if self.device == 'cpu' or not torch.cuda.is_available():
            return torch.float32

        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
        }
        return dtype_map.get(self.model_settings.torch_dtype, torch.float16)
