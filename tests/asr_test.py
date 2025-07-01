from jiwer import cer, wer
from sentence_transformers import SentenceTransformer, util

from src.core.pipe import Pipe
from tests.test_utils import validate_asr_results


class TestASRFunctionality:
    """Functional tests for ASR"""

    def test_single_audio_transcription(self, pipe_warmed: Pipe, test_audio_files: list[dict]) -> None:
        """Test transcription of a single audio file"""
        audio_file = test_audio_files[0]['audio']

        # Transcription
        results = pipe_warmed([audio_file])

        # Все проверки в одной функции
        validate_asr_results(results)

    def test_batch_audio_transcription(self, pipe_warmed: Pipe, test_audio_files: list[dict]) -> None:
        """Test batch transcription of multiple audio files"""
        # Batch transcription
        results = pipe_warmed([file['audio'] for file in test_audio_files])

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

    def test_quality_of_transcription(self, pipe_warmed: Pipe, test_audio_files: list[dict]) -> None:
        """Test quality of transcription"""
        score_model = SentenceTransformer('all-MiniLM-L6-v2')
        results = pipe_warmed([file['audio'] for file in test_audio_files])

        validate_asr_results(results, expected_count=len(test_audio_files))

        for result, file in zip(results, test_audio_files):
            result_text = ' '.join([chunk.text for chunk in result])
            reference_text = ' '.join([chunk.text for chunk in file['text']])
            # Using BERTScore for semantic similarity instead of exact matching

            # Calculate semantic similarity using embeddings
            embedding1 = score_model.encode(result_text, convert_to_tensor=True)
            embedding2 = score_model.encode(reference_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            assert similarity > 0.9, f'Semantic similarity too low: {similarity}'

            # Calculate Character Error Rate (CER) and Word Error Rate (WER)

            char_error_rate = cer(reference_text, result_text)
            word_error_rate = wer(reference_text, result_text)

            print(f'CER: {char_error_rate:.4f}, WER: {word_error_rate:.4f}')

            # Assert reasonable error rates
            assert char_error_rate < 0.3, f'Character Error Rate too high: {char_error_rate:.4f}'
            assert word_error_rate < 0.4, f'Word Error Rate too high: {word_error_rate:.4f}'
