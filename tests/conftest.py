import json
import os
from pathlib import Path

import librosa
import numpy as np
import pytest

from src.core.config import ASRConfig, AudioProcessorConfig, ModelConfig
from src.core.pipe import Pipe
from src.core.replica import Replica


@pytest.fixture(scope='session')
def test_audio_files() -> list[dict[str, np.ndarray | list[Replica]]]:
    """Fixture with paths to test audio files"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')

    with open(os.path.join(test_dir, 'whisper_audio.json'), 'r') as f:
        whisper_text = [Replica(**item) for item in json.load(f)]

    with open(os.path.join(test_dir, 'output.json'), 'r') as f:
        output_text = [Replica(**item) for item in json.load(f)]

    return [
        {'audio': librosa.load(os.path.join(test_dir, 'output.wav'), sr=16_000)[0], 'text': output_text},
        {'audio': librosa.load(os.path.join(test_dir, 'whisper_audio.wav'), sr=16_000)[0], 'text': whisper_text},
    ]


@pytest.fixture(scope='session')
def asr_config() -> ASRConfig:
    """Fixture for creating ASR config"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo')


@pytest.fixture(scope='session')
def processor_config() -> AudioProcessorConfig:
    """Fixture for creating processor config"""
    return AudioProcessorConfig()


@pytest.fixture(scope='session')
def pipe(asr_config: ASRConfig, processor_config: AudioProcessorConfig) -> Pipe:
    """Fixture for creating ASR model - initialized once per session"""
    return Pipe(config_asr=asr_config, config_processor=processor_config)


@pytest.fixture(scope='session')
def pipe_warmed(asr_config: ASRConfig, processor_config: AudioProcessorConfig) -> Pipe:
    """Fixture for creating warmed up ASR model - initialized once per session"""
    pipe = Pipe(config_asr=asr_config, config_processor=processor_config)
    pipe.warmup()
    return pipe


@pytest.fixture(scope='session')
def benchmark_results_dir() -> Path:
    """Fixture for benchmark results directory"""
    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture(scope='session')
def cpu_config() -> ASRConfig:
    """Fixture for CPU configuration"""
    return ASRConfig(
        model_id='openai/whisper-large-v3-turbo', model_settings=ModelConfig(torch_dtype='float32', device='cpu')
    )


@pytest.fixture(scope='session')
def cpu_pipe(cpu_config: ASRConfig, processor_config: AudioProcessorConfig) -> Pipe:
    """Fixture for creating CPU ASR model - initialized once per session"""
    return Pipe(config_asr=cpu_config, config_processor=processor_config)
