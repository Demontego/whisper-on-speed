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

    def warmup(self) -> None:
        """
        Warm up the pipeline
        """
        self.asr.warmup(1, self.processor.sr)
        self.is_warmed_up = True

    def __call__(self, audios: list[np.ndarray]) -> list[list[Replica]]:
        """
        Process the audio

        Args:
            audios: List of audio arrays

        Returns:
            List of lists of replicas
        """
        all_seconds = []
        all_input_features = []
        masks = []
        total_audio_len = len(audios)
        for ind, audio in enumerate(audios):
            input_features, seconds = self.processor.preprocess(audio)
            all_seconds.append(seconds)
            all_input_features.append(input_features)
            masks.extend([ind] * len(seconds))
        masks = np.array(masks)
        inputs = np.concatenate(all_input_features, axis=0)
        tokens = self.asr(inputs, self.processor.sr)

        results = []

        for ind in range(total_audio_len):
            mask = masks == ind
            tokens_ind = tokens[mask]
            seconds_ind = all_seconds[ind]
            result = self.processor.postprocess(tokens_ind, seconds_ind)
            results.append(result)

        return results
