from src.core.asr import ASRChunk, ASRonSPEED
from src.core.config import ASRConfig


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
        audio_file = test_audio_files[0]

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
        results = asr_model.process_batch(test_audio_files)

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
        asr_model.warmup(num_warmup_steps=1)

        # Check that model is warmed up after warmup
        assert asr_model.is_warmed_up

    def test_empty_audio_handling(self, asr_model):
        """Test handling of empty or very short audio"""
        import numpy as np

        # Very short audio (less than a second)
        short_audio = np.zeros(8000)  # 0.5 seconds at 16kHz

        result = asr_model.process_audio(short_audio)

        # Should still return a list (might be empty or contain empty text)
        assert isinstance(result, list)

    def test_different_config_parameters(self):
        """Test ASR with different configuration parameters"""
        # Test with different chunk size
        config = ASRConfig(model_id='openai/whisper-large-v3-turbo', seconds_per_chunk=5, batch_size=8)
        model = ASRonSPEED(config=config)

        assert model.config.seconds_per_chunk == 5
        assert model.config.batch_size == 8
        assert not model.is_warmed_up

    def test_device_selection(self):
        """Test device selection logic"""
        import torch

        # Test CPU config
        cpu_config = ASRConfig(device='cpu')
        cpu_model = ASRonSPEED(config=cpu_config)
        assert cpu_model.device.type == 'cpu'

        # Test CUDA config (if available)
        if torch.cuda.is_available():
            cuda_config = ASRConfig(device='cuda')
            cuda_model = ASRonSPEED(config=cuda_config)
            assert cuda_model.device.type == 'cuda'

    def test_model_dtype_configuration(self):
        """Test model dtype configuration"""
        import torch

        # Test float32 configuration
        config = ASRConfig()
        config.model_settings.torch_dtype = 'float32'

        expected_dtype = torch.float32 if config.device == 'cpu' or not torch.cuda.is_available() else torch.float32
        actual_dtype = config.get_torch_dtype()

        assert actual_dtype == expected_dtype

    def test_generation_config_parameters(self, asr_model):
        """Test that generation config parameters are properly set"""
        assert asr_model.generate_kwargs is not None
        assert 'max_new_tokens' in asr_model.generate_kwargs
        assert 'num_beams' in asr_model.generate_kwargs
        assert 'temperature' in asr_model.generate_kwargs

    def test_chunk_processing_logic(self, asr_model, test_audio_files):
        """Test that chunk processing works correctly"""
        audio_file = test_audio_files[0]
        result = asr_model.process_audio(audio_file)

        # Check that chunks have proper structure
        for chunk in result:
            assert isinstance(chunk, ASRChunk)
            assert isinstance(chunk.text, str)
            assert chunk.start_time is None or isinstance(chunk.start_time, (int, float))
            assert chunk.end_time is None or isinstance(chunk.end_time, (int, float))

            # If both timestamps exist, start should be before end
            if chunk.start_time is not None and chunk.end_time is not None:
                assert chunk.start_time <= chunk.end_time
