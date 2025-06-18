# Whisper on Speed

Оптимизированная библиотека для быстрого распознавания речи на базе OpenAI Whisper с поддержкой GPU и батчевой обработки.

## Установка

```bash
# Установка зависимостей
uv sync

# Или для разработки
uv sync --dev
```

## Использование

```python
from src.core.asr import ASRonSPEED
import librosa

# Инициализация модели
asr = ASRonSPEED(model_id="openai/whisper-large-v3-turbo")

# Опционально: прогрев модели для лучшей производительности
asr.warmup()

# Загрузка аудио
audio, _ = librosa.load("audio.wav", sr=16000)

# Транскрипция одного файла
chunks = asr.procces_audio(audio)
for chunk in chunks:
    print(f"{chunk.start_time:.2f}s - {chunk.end_time:.2f}s: {chunk.text}")

# Батчевая обработка
audio_files = [audio1, audio2, audio3]
batch_results = asr.process_batch(audio_files)
```

## Особенности

- 🚀 Оптимизированная обработка с использованием GPU
- 📦 Поддержка батчевой обработки
- ⏱️ Возвращение временных меток для каждого сегмента
- 🔥 Функция прогрева для стабильной производительности
- 📊 Встроенные бенчмарки

## Тестирование

```bash
# Запуск всех тестов
uv run pytest

# Только бенчмарки
uv run pytest -m benchmark
```

## Требования

- Python 3.13+
- CUDA-совместимая GPU (рекомендуется)
- uv для управления зависимостями
