import json
import os
from pathlib import Path

import librosa
import numpy as np
import pytest

from src.core.asr import ASRChunk, ASRonSPEED
from src.core.config import ASRConfig


@pytest.fixture(scope='session')
def test_audio_files() -> list[dict[str, np.ndarray | list[ASRChunk]]]:
    """Fixture with paths to test audio files"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')

    with open(os.path.join(test_dir, 'whisper_audio.json'), 'r') as f:
        whisper_text = [ASRChunk(**item) for item in json.load(f)]

    with open(os.path.join(test_dir, 'output.json'), 'r') as f:
        output_text = [ASRChunk(**item) for item in json.load(f)]

    return [
        {'audio': librosa.load(os.path.join(test_dir, 'output.wav'), sr=16_000)[0], 'text': output_text},
        {'audio': librosa.load(os.path.join(test_dir, 'whisper_audio.wav'), sr=16_000)[0], 'text': whisper_text},
    ]


@pytest.fixture(scope='session')
def asr_config() -> ASRConfig:
    """Fixture for creating ASR config"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo')


@pytest.fixture(scope='session')
def asr_model(asr_config) -> ASRonSPEED:
    """Fixture for creating ASR model - initialized once per session"""
    return ASRonSPEED(config=asr_config)


@pytest.fixture(scope='session')
def asr_model_warmed() -> ASRonSPEED:
    """Fixture for creating warmed up ASR model - initialized once per session"""
    config = ASRConfig(model_id='openai/whisper-large-v3-turbo')
    model = ASRonSPEED(config=config)
    model.warmup(num_warmup_steps=2)
    return model


@pytest.fixture(scope='session')
def benchmark_results_dir() -> Path:
    """Fixture for benchmark results directory"""
    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture(scope='session')
def cpu_config() -> ASRConfig:
    """Fixture for CPU configuration"""
    return ASRConfig(model_id='openai/whisper-large-v3-turbo', device='cpu')
