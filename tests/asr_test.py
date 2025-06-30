import difflib

from src.core.pipe import Pipe
from tests.test_utils import validate_asr_results


class TestASRFunctionality:
    """Functional tests for ASR"""

    def test_single_audio_transcription(self, pipe: Pipe, test_audio_files: list[dict]) -> None:
        """Test transcription of a single audio file"""
        audio_file = test_audio_files[0]['audio']

        # Transcription
        results = pipe([audio_file])

        # Все проверки в одной функции
        validate_asr_results(results)

    def test_batch_audio_transcription(self, pipe: Pipe, test_audio_files: list[dict]) -> None:
        """Test batch transcription of multiple audio files"""
        # Batch transcription
        results = pipe([file['audio'] for file in test_audio_files])

        # Все проверки в одной функции
        validate_asr_results(results, expected_count=len(test_audio_files))

    def test_warmup_functionality(self, pipe: Pipe) -> None:
        """Test warmup functionality"""
        # Check that model is not warmed up initially
        assert not pipe.is_warmed_up

        # Model warmup
        pipe.warmup()

        # Check that model is warmed up after warmup
        assert pipe.is_warmed_up

    def test_quality_of_transcription(self, pipe: Pipe, test_audio_files: list[dict]) -> None:
        """Test quality of transcription"""
        results = pipe([file['audio'] for file in test_audio_files])

        validate_asr_results(results, expected_count=len(test_audio_files))

        for result, file in zip(results, test_audio_files):
            result_text = ' '.join([chunk.text for chunk in result])
            reference_text = ' '.join([chunk.text for chunk in file['text']])

            similarity = difflib.SequenceMatcher(None, result_text, reference_text).ratio()
            assert similarity > 0.9, f'Similarity too low: {similarity}'
