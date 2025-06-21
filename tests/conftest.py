import os
from pathlib import Path

import librosa
import pytest

from src.core.asr import ASRonSPEED
from src.core.config import ASRConfig


@pytest.fixture
def test_audio_files():
    """Fixture with paths to test audio files"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    return [
        librosa.load(os.path.join(test_dir, 'output.wav'), sr=16_000)[0],
        librosa.load(os.path.join(test_dir, 'whisper_audio.wav'), sr=16_000)[0],
    ]


@pytest.fixture
def asr_config():
    """Fixture for creating ASR config"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo')


@pytest.fixture
def asr_model(asr_config):
    """Fixture for creating ASR model"""
    return ASRonSPEED(config=asr_config)


@pytest.fixture
def asr_model_warmed(asr_config):
    """Fixture for creating warmed up ASR model"""
    model = ASRonSPEED(config=asr_config)
    model.warmup(num_warmup_steps=2)
    return model


@pytest.fixture
def benchmark_results_dir():
    """Fixture for benchmark results directory"""
    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture
def cpu_config():
    """Fixture for CPU configuration"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo', device='cpu')


@pytest.fixture
def cuda_config():
    """Fixture for CUDA configuration"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo', device='cuda')
