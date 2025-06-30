import logging

import numpy as np

from src.core.asr import ASRonSPEED
from src.core.config import ASRConfig, AudioProcessorConfig
from src.core.processor import AudioProcessor
from src.core.replica import Replica


class Pipe:
    def __init__(self, config_asr: ASRConfig, config_processor: AudioProcessorConfig) -> None:
        """
        Initialize the pipeline

        Args:
            config_asr: ASR configuration
            config_processor: Audio processor configuration
        """
        self.asr = ASRonSPEED(config_asr)
        self.processor = AudioProcessor(config_processor)
        self.config_asr = config_asr
        self.is_warmed_up = False
        self.logger = logging.getLogger(__name__)

    def warmup(self) -> None:
        """
        Warm up the ASR model with a dummy input.

        Raises:
            RuntimeError: If warmup fails
        """
        try:
            self.asr.warmup(1, self.processor.sr)
            self.is_warmed_up = True
        except Exception as e:
            self.logger.error(f'Warmup failed: {e}')
            raise RuntimeError(f'Failed to warm up ASR model: {e}') from e

    def __call__(self, audios: list[np.ndarray]) -> list[list[Replica]]:
        """
        Process multiple audio inputs and return timestamped transcriptions.

        Args:
            audios: List of 1D numpy arrays containing audio samples

        Returns:
            List of transcription results, one per input audio

        Raises:
            ValueError: If audios is empty or contains invalid data
            RuntimeError: If model is not warmed up
        """
        if not audios:
            raise ValueError('No audio inputs provided')
        if not self.is_warmed_up:
            raise RuntimeError('Model must be warmed up before inference')

        # Preprocess all audios in one loop
        all_seconds = []
        all_input_features = []
        audio_indices = []

        # Use list comprehension for preprocessing
        preprocessed_data = [self.processor.preprocess(audio) for audio in audios]

        # Unpack the results
        for i, (input_features, seconds) in enumerate(preprocessed_data):
            all_input_features.append(input_features)
            all_seconds.append(seconds)
            audio_indices.extend([i] * len(seconds))

        # Convert to numpy array once
        audio_indices = np.array(audio_indices)
        inputs = np.concatenate(all_input_features, axis=0)

        # Process all inputs in a single batch
        tokens = self.asr(inputs, self.processor.sr)

        # Use list comprehension for postprocessing
        results = [self.processor.postprocess(tokens[audio_indices == i], all_seconds[i]) for i in range(len(audios))]

        return results
