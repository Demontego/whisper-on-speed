import difflib

from tests.test_utils import validate_asr_results


class TestASRFunctionality:
    """Functional tests for ASR"""

    def test_model_initialization(self, asr_model):
        """Test model initialization"""
        assert not asr_model.is_warmed_up
        assert asr_model.config.model_id == 'openai/whisper-large-v3-turbo'
        assert asr_model.device is not None
        assert asr_model.pipe is not None

    def test_config_creation(self, asr_config):
        """Test configuration creation"""
        assert asr_config.model_id == 'openai/whisper-large-v3-turbo'
        assert asr_config.seconds_per_chunk == 10
        assert asr_config.batch_size == 16
        assert asr_config.sample_rate == 16_000

    def test_single_audio_transcription(self, asr_model, test_audio_files):
        """Test transcription of a single audio file"""
        audio_file = test_audio_files[0]['audio']

        # Transcription
        result = asr_model.process_audio(audio_file)

        # Все проверки в одной функции
        validate_asr_results(result)

    def test_batch_audio_transcription(self, asr_model, test_audio_files):
        """Test batch transcription of multiple audio files"""
        # Batch transcription
        results = asr_model.process_batch([file['audio'] for file in test_audio_files])

        # Все проверки в одной функции
        validate_asr_results(results, expected_count=len(test_audio_files))

    def test_warmup_functionality(self, asr_model):
        """Test warmup functionality"""
        # Check that model is not warmed up initially
        assert not asr_model.is_warmed_up

        # Model warmup
        asr_model.warmup(num_warmup_steps=3)

        # Check that model is warmed up after warmup
        assert asr_model.is_warmed_up

    def test_quality_of_transcription(self, asr_model, test_audio_files):
        """Test quality of transcription"""
        for file in test_audio_files:
            result = asr_model.process_audio(file['audio'])

            validate_asr_results(result)

            result_text = ' '.join([chunk.text for chunk in result])
            reference_text = ' '.join([chunk.text for chunk in file['text']])

            similarity = difflib.SequenceMatcher(None, result_text, reference_text).ratio()
            assert similarity > 0.9, f'Similarity too low: {similarity}'
