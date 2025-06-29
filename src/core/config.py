import torch
from pydantic import BaseModel, Field


class AudioProcessorConfig(BaseModel):
    model_id: str = Field(default='openai/whisper-large-v3-turbo', description='model id')
    model_vad: str = Field(default='snakers4/silero-vad', description='model vad')
    sr: int = Field(default=16_000, description='sample rate')
    chunk_sec: int = Field(default=30, description='chunk seconds')
    overlap_sec: float = Field(default=0.3, description='overlap seconds')


class ModelConfig(BaseModel):
    """Configuration for model loading"""

    torch_dtype: str = 'float16'
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    device: str = Field(default='cuda', description='device')


class GenerationConfig(BaseModel):
    """Configuration for text generation"""

    max_new_tokens: int = Field(default=100, description='maximum number of new tokens')
    num_beams: int = Field(default=3, description='number of beams for beam search')
    condition_on_prev_tokens: bool = Field(default=True, description='consider previous tokens')
    length_penalty: float = Field(default=1.0, description='length penalty')
    temperature: float = Field(default=0.1, description='temperature for generation')
    logprob_threshold: float = Field(default=0.6, description='log probability threshold')
    is_multilingual: bool = Field(default=True, description='is multilingual')
    task: str = Field(default='transcribe', description='task')
    return_timestamps: bool = Field(default=True, description='return timestamps')


class ASRConfig(BaseModel):
    """Основная конфигурация для ASR"""

    model_id: str = Field(default='openai/whisper-large-v3-turbo', description='model id')
    batch_size: int = Field(default=178, description='batch size')
    warmup_steps: int = Field(default=3, description='number of warmup steps')
    pad_value: int = Field(default=50257, description='pad value')

    # Renamed from model_config to avoid conflict with Pydantic's reserved attribute
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string representation"""
        # For CPU, force float32, as float16 is inefficient
        if self.model_settings.device == 'cpu' or not torch.cuda.is_available():
            return torch.float32

        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
        }
        return dtype_map.get(self.model_settings.torch_dtype, torch.float16)
