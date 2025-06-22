import difflib

from src.core.asr import ASRChunk


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

        # Result checks
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], ASRChunk)
        assert result[0].text is not None
        assert len(result[0].text) > 0

        # Check timestamps if they exist
        if result[0].start_time is not None and result[0].end_time is not None:
            assert result[0].start_time >= 0
            assert result[0].end_time > result[0].start_time

    def test_batch_audio_transcription(self, asr_model, test_audio_files):
        """Test batch transcription of multiple audio files"""
        # Batch transcription
        results = asr_model.process_batch([file['audio'] for file in test_audio_files])

        # Results checks
        assert isinstance(results, list)
        assert len(results) == len(test_audio_files)

        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], ASRChunk)
            assert result[0].text is not None
            assert len(result[0].text) > 0

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
            assert result[0].text is not None
            assert len(result) == len(file['text'])
            for i in range(len(result)):
                assert difflib.SequenceMatcher(None, result[i].text, file['text'][i].text).ratio() > 0.8
                if result[i].start_time is not None and file['text'][i].start_time is not None:
                    assert abs(result[i].start_time - file['text'][i].start_time) < 1.0
                if result[i].end_time is not None and file['text'][i].end_time is not None:
                    assert abs(result[i].end_time - file['text'][i].end_time) < 1.0
