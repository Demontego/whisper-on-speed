# Whisper on Speed

Optimized library for fast speech recognition based on OpenAI Whisper with GPU support and batch processing.

## Installation

```bash
# Install dependencies
uv sync

# Or for development
uv sync --dev
```

## Usage

```python
from src.core.asr import ASRonSPEED
import librosa

# Initialize model
asr = ASRonSPEED(model_id="openai/whisper-large-v3-turbo")

# Optional: warm up model for better performance
asr.warmup()

# Load audio
audio, _ = librosa.load("audio.wav", sr=16000)

# Transcribe single file
chunks = asr.process_audio(audio)
for chunk in chunks:
    print(f"{chunk.start_time:.2f}s - {chunk.end_time:.2f}s: {chunk.text}")

# Batch processing
audio_files = [audio1, audio2, audio3]
batch_results = asr.process_batch(audio_files)
```

## Features

- 🚀 Optimized processing with GPU acceleration
- 📦 Batch processing support
- ⏱️ Timestamp return for each segment
- 🔥 Warmup function for stable performance
- 📊 Built-in benchmarks

## Testing

```bash
# Run all tests
uv run pytest

# Benchmarks only
uv run pytest -m benchmark
```

## Requirements

- Python 3.13+
- CUDA-compatible GPU (recommended)
- uv for dependency management
