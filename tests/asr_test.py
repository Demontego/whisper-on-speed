import datetime
import json
import os
import time
from pathlib import Path

import librosa
import pytest
import torch

from src.core.asr import ASRChunk, ASRonSPEED


def get_gpu_info():
    """Get GPU information for benchmarking"""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "device_capability": None,
        "memory_info": None,
        "driver_version": None,
        "torch_version": torch.__version__
    }
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_info.update({
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "device_capability": torch.cuda.get_device_capability(current_device),
            "memory_info": {
                "total_memory": torch.cuda.get_device_properties(current_device).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(current_device),
                "memory_reserved": torch.cuda.memory_reserved(current_device),
                "max_memory_allocated": torch.cuda.max_memory_allocated(current_device),
                "max_memory_reserved": torch.cuda.max_memory_reserved(current_device)
            }
        })
        
        # Try to get CUDA driver version
        try:
            gpu_info["driver_version"] = torch.version.cuda  # type: ignore
        except:
            gpu_info["driver_version"] = "unknown"
    
    return gpu_info


class TestWhisperASR:
    """Tests for WhisperASR"""
    
    @pytest.fixture
    def asr_model(self):
        """Fixture for creating ASR model"""
        return ASRonSPEED(model_id="openai/whisper-large-v3-turbo")
    
    @pytest.fixture
    def test_audio_files(self):
        """Fixture with paths to test audio files"""
        test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        return [
            librosa.load(os.path.join(test_dir, "output.wav"), sr=16_000)[0],
            librosa.load(os.path.join(test_dir, "whisper_audio.wav"), sr=16_000)[0],
        ]
    
    def test_model_initialization(self, asr_model):
        """Test model initialization"""
        assert not asr_model.is_warmed_up
    
    def test_single_audio_transcription(self, asr_model, test_audio_files):
        """Test transcription of a single audio file"""
        # Check that the file exists
        audio_file = test_audio_files[0]
        
        # Transcription
        result = asr_model.process_audio(audio_file)
        
        # Result checks
        assert isinstance(result, list)
        assert isinstance(result[0], ASRChunk)
        assert result[0].text is not None
        assert len(result[0].text) > 0
        
        # Check timestamps if they exist
        assert result[0].start_time is not None
        assert result[0].end_time is not None
    
    def test_batch_audio_transcription(self, asr_model, test_audio_files):
        """Test batch transcription of multiple audio files"""
        # Batch transcription
        results = asr_model.process_batch(test_audio_files)
        
        # Results checks
        assert isinstance(results, list)
        assert len(results) == len(test_audio_files)
        
        for result in results:
            assert isinstance(result, list)
            assert isinstance(result[0], ASRChunk)
            assert result[0].text is not None
            assert len(result[0].text) > 0
    
    def test_warmup_functionality(self, asr_model, test_audio_files):
        """Test warmup functionality"""
        # Check that model is not warmed up
        assert not asr_model.is_warmed_up
        
        if test_audio_files:
            # Model warmup  
            asr_model.warmup(num_warmup_steps=1)
            
            # Check that model is warmed up
            assert asr_model.is_warmed_up
        else:
            pytest.skip("No available audio files for warmup")


class TestWhisperASRBenchmark:
    """Benchmark tests for WhisperASR"""
    
    @pytest.fixture
    def asr_model_warmed(self, test_audio_files):
        """Fixture for creating warmed up ASR model"""
        model = ASRonSPEED(model_id="openai/whisper-large-v3-turbo")
        model.warmup(num_warmup_steps=2)
        return model
    
    @pytest.fixture
    def test_audio_files(self):
        """Fixture with paths to test audio files"""
        test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        return [
            librosa.load(os.path.join(test_dir, "output.wav"), sr=16_000)[0],
            librosa.load(os.path.join(test_dir, "whisper_audio.wav"), sr=16_000)[0],
        ]
    
    @pytest.fixture
    def benchmark_results_dir(self):
        """Fixture for benchmark results directory"""
        results_dir = Path(__file__).parent / "benchmark_results"
        results_dir.mkdir(exist_ok=True)
        return results_dir
    
    def _save_benchmark_results(self, benchmark_results_dir, test_name, results_data):
        """Save benchmark results to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = benchmark_results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Benchmark results saved to: {filepath}")
    
    @pytest.mark.benchmark
    def test_single_transcription_performance(self, asr_model_warmed, test_audio_files, benchmark_results_dir):
        """Benchmark test for single file transcription performance"""
        # Get GPU info before benchmark
        gpu_info = get_gpu_info()
        
        # Measure execution time
        num_runs = 5
        times = []
        audio_duration = len(test_audio_files[0]) / 16000  # Duration in seconds
        
        for _ in range(num_runs):
            start_time = time.time()
            result = asr_model_warmed.process_audio(test_audio_files[0])
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Check that result is correct
            assert isinstance(result, list)
            assert isinstance(result[0], ASRChunk)
            assert result[0].text is not None
            assert len(result[0].text) > 0
        
        # Performance statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate audio processing metrics
        avg_rtf = audio_duration / avg_time  # Audio seconds per processing second
        min_rtf = audio_duration / max_time  # Best case (min processing time)
        max_rtf = audio_duration / min_time  # Worst case (max processing time)
        
        # Prepare results data
        results_data = {
            "test_name": "single_transcription_performance",
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": "openai/whisper-large-v3-turbo",
            "num_runs": num_runs,
            "execution_times": times,
            "statistics": {
                "average_time": avg_time,
                "minimum_time": min_time,
                "maximum_time": max_time,
                "std_deviation": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            },
            "rtf_metrics": {
                "average_rtf": avg_rtf,
                "minimum_rtf": min_rtf,
                "maximum_rtf": max_rtf,
                "avg_audio_seconds_per_processing_second": avg_rtf,
                "processing_faster_than_realtime": avg_rtf > 1.0
            },
            "audio_info": {
                "sample_rate": 16000,
                "duration_seconds": audio_duration
            },
            "hardware_info": {
                "gpu": gpu_info
            }
        }
        
        # Save results
        self._save_benchmark_results(benchmark_results_dir, "single_transcription", results_data)
        
        print(f"\n=== Single Transcription Benchmark ===")
        print(f"GPU: {gpu_info['device_name'] if gpu_info['cuda_available'] else 'CPU only'}")
        if gpu_info['cuda_available']:
            memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
            print(f"GPU Memory: {memory_gb:.1f}GB")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Number of runs: {num_runs}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Minimum time: {min_time:.3f}s")
        print(f"Maximum time: {max_time:.3f}s")
        print(f"Std deviation: {results_data['statistics']['std_deviation']:.3f}s")
        print(f"Average RTF: {avg_rtf:.2f}x (аудио секунд за секунду обработки)")
        print(f"RTF range: {min_rtf:.2f}x - {max_rtf:.2f}x")
        print(f"Processing {'faster' if avg_rtf > 1.0 else 'slower'} than real-time")
        
        # Check that time is reasonable (no more than 30 seconds)
        assert avg_time < 30.0, f"Transcription too slow: {avg_time:.2f}s"
    
    @pytest.mark.benchmark
    def test_batch_transcription_performance(self, asr_model_warmed, test_audio_files, benchmark_results_dir):
        """Benchmark test for batch transcription performance"""
        # Get GPU info before benchmark
        gpu_info = get_gpu_info()
        
        # Measure execution time
        num_runs = 3
        times = []
        total_audio_duration = sum(len(audio) / 16000 for audio in test_audio_files)
        
        for _ in range(num_runs):
            start_time = time.time()
            results = asr_model_warmed.process_batch(test_audio_files)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Check that results are correct
            assert isinstance(results, list)
            assert len(results) == len(test_audio_files)
            
            for result in results:
                assert isinstance(result, list)
                assert isinstance(result[0], ASRChunk)
                assert result[0].text is not None
                assert len(result[0].text) > 0
        
        # Performance statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_time_per_file = avg_time / len(test_audio_files)
        
        # Calculate audio processing metrics
        avg_rtf = total_audio_duration / avg_time
        min_rtf = total_audio_duration / max_time
        max_rtf = total_audio_duration / min_time
        avg_rtf_per_file = (total_audio_duration / len(test_audio_files)) / avg_time_per_file
        
        # Prepare results data
        results_data = {
            "test_name": "batch_transcription_performance",
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": "openai/whisper-large-v3-turbo",
            "num_runs": num_runs,
            "num_files": len(test_audio_files),
            "execution_times": times,
            "statistics": {
                "average_time": avg_time,
                "minimum_time": min_time,
                "maximum_time": max_time,
                "average_time_per_file": avg_time_per_file,
                "std_deviation": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            },
            "rtf_metrics": {
                "average_rtf": avg_rtf,
                "minimum_rtf": min_rtf,
                "maximum_rtf": max_rtf,
                "average_rtf_per_file": avg_rtf_per_file,
                "total_audio_seconds_per_processing_second": avg_rtf,
                "processing_faster_than_realtime": avg_rtf > 1.0
            },
            "audio_info": {
                "sample_rate": 16000,
                "total_files": len(test_audio_files),
                "total_audio_duration": total_audio_duration,
                "file_durations": [len(audio) / 16000 for audio in test_audio_files]
            },
            "hardware_info": {
                "gpu": gpu_info
            }
        }
        
        # Save results
        self._save_benchmark_results(benchmark_results_dir, "batch_transcription", results_data)
        
        print(f"\n=== Batch Transcription Benchmark ===")
        print(f"GPU: {gpu_info['device_name'] if gpu_info['cuda_available'] else 'CPU only'}")
        if gpu_info['cuda_available']:
            memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
            print(f"GPU Memory: {memory_gb:.1f}GB")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
        print(f"Number of files: {len(test_audio_files)}")
        print(f"Number of runs: {num_runs}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Time per file: {avg_time_per_file:.3f}s")
        print(f"Minimum time: {min_time:.3f}s")
        print(f"Maximum time: {max_time:.3f}s")
        print(f"Std deviation: {results_data['statistics']['std_deviation']:.3f}s")
        print(f"Average RTF: {avg_rtf:.2f}x (аудио секунд за секунду обработки)")
        print(f"RTF per file: {avg_rtf_per_file:.2f}x")
        print(f"RTF range: {min_rtf:.2f}x - {max_rtf:.2f}x")
        print(f"Processing {'faster' if avg_rtf > 1.0 else 'slower'} than real-time")
        
        # Check that time is reasonable
        assert avg_time < 60.0, f"Batch transcription too slow: {avg_time:.2f}s"
    
    @pytest.mark.benchmark
    def test_warmup_vs_cold_performance(self, test_audio_files, benchmark_results_dir):
        """Benchmark comparison of performance with and without warmup"""
        # Get GPU info before benchmark
        gpu_info = get_gpu_info()
        
        audio_duration = len(test_audio_files[0]) / 16000
        
        # Test without warmup
        cold_model = ASRonSPEED(model_id="openai/whisper-large-v3-turbo")
        start_time = time.time()
        cold_result = cold_model.process_audio(test_audio_files[0])
        cold_time = time.time() - start_time
        
        # Test with warmup
        warm_model = ASRonSPEED(model_id="openai/whisper-large-v3-turbo")
        warm_model.warmup(num_warmup_steps=2)
        
        start_time = time.time()
        warm_result = warm_model.process_audio(test_audio_files[0])
        warm_time = time.time() - start_time
        
        # Check results
        assert isinstance(cold_result, list)
        assert isinstance(warm_result, list)
        assert cold_result[0].text is not None
        assert warm_result[0].text is not None
        
        # Calculate speedup and RTF
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        cold_rtf = audio_duration / cold_time
        warm_rtf = audio_duration / warm_time
        
        # Prepare results data
        results_data = {
            "test_name": "warmup_vs_cold_performance",
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": "openai/whisper-large-v3-turbo",
            "results": {
                "cold_start_time": cold_time,
                "warmed_up_time": warm_time,
                "speedup": speedup,
                "warmup_steps": 2
            },
            "rtf_metrics": {
                "cold_rtf": cold_rtf,
                "warm_rtf": warm_rtf,
                "rtf_improvement": warm_rtf / cold_rtf if cold_rtf > 0 else float('inf'),
                "cold_faster_than_realtime": cold_rtf > 1.0,
                "warm_faster_than_realtime": warm_rtf > 1.0
            },
            "audio_info": {
                "sample_rate": 16000,
                "duration_seconds": audio_duration
            },
            "hardware_info": {
                "gpu": gpu_info
            }
        }
        
        # Save results
        self._save_benchmark_results(benchmark_results_dir, "warmup_comparison", results_data)
        
        print(f"\n=== Warmup vs Cold Performance Comparison ===")
        print(f"GPU: {gpu_info['device_name'] if gpu_info['cuda_available'] else 'CPU only'}")
        if gpu_info['cuda_available']:
            memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
            print(f"GPU Memory: {memory_gb:.1f}GB")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Cold start time: {cold_time:.3f}s")
        print(f"Warmed up time: {warm_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Cold RTF: {cold_rtf:.2f}x (аудио секунд за секунду обработки)")
        print(f"Warm RTF: {warm_rtf:.2f}x (аудио секунд за секунду обработки)")
        print(f"RTF improvement: {results_data['rtf_metrics']['rtf_improvement']:.2f}x")
        print(f"Cold processing {'faster' if cold_rtf > 1.0 else 'slower'} than real-time")
        print(f"Warm processing {'faster' if warm_rtf > 1.0 else 'slower'} than real-time")
        
        # Usually warmup should speed up subsequent calls
        # But for the first call it may not always be noticeable
    
    @pytest.mark.benchmark
    def test_cpu_vs_cuda_performance(self, test_audio_files, benchmark_results_dir):
        """Benchmark comparison of CPU vs CUDA performance"""
        # Get GPU info before benchmark
        gpu_info = get_gpu_info()
        
        audio_duration = len(test_audio_files[0]) / 16000
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CPU vs CUDA comparison")
        
        print(f"\n=== CPU vs CUDA Performance Comparison ===")
        print(f"Audio duration: {audio_duration:.2f}s")
        
        # Test with CPU
        print("Testing CPU performance...")
        cpu_model = ASRonSPEED(model_id="openai/whisper-large-v3-turbo", device='cpu')
        cpu_model.warmup(num_warmup_steps=1)  # Minimal warmup for CPU
        
        cpu_times = []
        num_runs = 3  # Fewer runs for CPU as it's typically slower
        
        for i in range(num_runs):
            print(f"  CPU run {i+1}/{num_runs}...")
            start_time = time.time()
            cpu_result = cpu_model.process_audio(test_audio_files[0])
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)
            
            # Check result validity
            assert isinstance(cpu_result, list)
            assert len(cpu_result) > 0
            assert cpu_result[0].text is not None
        
        cpu_avg_time = sum(cpu_times) / len(cpu_times)
        cpu_rtf = audio_duration / cpu_avg_time
        
        # Test with CUDA  
        print("Testing CUDA performance...")
        cuda_model = ASRonSPEED(model_id="openai/whisper-large-v3-turbo", device='cuda')
        cuda_model.warmup(num_warmup_steps=2)
        
        cuda_times = []
        
        for i in range(num_runs):
            print(f"  CUDA run {i+1}/{num_runs}...")
            start_time = time.time()
            cuda_result = cuda_model.process_audio(test_audio_files[0])
            cuda_time = time.time() - start_time
            cuda_times.append(cuda_time)
            
            # Check result validity
            assert isinstance(cuda_result, list)
            assert len(cuda_result) > 0
            assert cuda_result[0].text is not None
        
        cuda_avg_time = sum(cuda_times) / len(cuda_times)
        cuda_rtf = audio_duration / cuda_avg_time
        
        # Calculate speedup and comparison metrics
        speedup = cpu_avg_time / cuda_avg_time if cuda_avg_time > 0 else float('inf')
        rtf_improvement = cuda_rtf / cpu_rtf if cpu_rtf > 0 else float('inf')
        
        # Performance statistics
        cpu_stats = {
            "average_time": cpu_avg_time,
            "minimum_time": min(cpu_times),
            "maximum_time": max(cpu_times),
            "std_deviation": (sum((t - cpu_avg_time) ** 2 for t in cpu_times) / len(cpu_times)) ** 0.5,
            "rtf": cpu_rtf
        }
        
        cuda_stats = {
            "average_time": cuda_avg_time,
            "minimum_time": min(cuda_times),
            "maximum_time": max(cuda_times),
            "std_deviation": (sum((t - cuda_avg_time) ** 2 for t in cuda_times) / len(cuda_times)) ** 0.5,
            "rtf": cuda_rtf
        }
        
        # Prepare results data
        results_data = {
            "test_name": "cpu_vs_cuda_performance",
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": "openai/whisper-large-v3-turbo",
            "num_runs": num_runs,
            "audio_info": {
                "sample_rate": 16000,
                "duration_seconds": audio_duration
            },
            "cpu_results": {
                "execution_times": cpu_times,
                "statistics": cpu_stats,
                "device": "cpu",
                "faster_than_realtime": cpu_rtf > 1.0
            },
            "cuda_results": {
                "execution_times": cuda_times,
                "statistics": cuda_stats,
                "device": "cuda",
                "faster_than_realtime": cuda_rtf > 1.0
            },
            "comparison": {
                "cuda_speedup": speedup,
                "rtf_improvement": rtf_improvement,
                "cpu_vs_cuda_time_ratio": cpu_avg_time / cuda_avg_time if cuda_avg_time > 0 else float('inf'),
                "winner": "cuda" if cuda_avg_time < cpu_avg_time else "cpu"
            },
            "hardware_info": {
                "gpu": gpu_info
            }
        }
        
        # Save results
        self._save_benchmark_results(benchmark_results_dir, "cpu_vs_cuda", results_data)
        
        # Print detailed comparison
        print(f"\n=== Results Summary ===")
        print(f"CPU Performance:")
        print(f"  Average time: {cpu_avg_time:.3f}s")
        print(f"  RTF: {cpu_rtf:.2f}x")
        print(f"  Faster than real-time: {'Yes' if cpu_rtf > 1.0 else 'No'}")
        
        print(f"\nCUDA Performance:")
        print(f"  GPU: {gpu_info['device_name']}")
        if gpu_info['cuda_available']:
            memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
            print(f"  GPU Memory: {memory_gb:.1f}GB")
        print(f"  Average time: {cuda_avg_time:.3f}s")
        print(f"  RTF: {cuda_rtf:.2f}x")
        print(f"  Faster than real-time: {'Yes' if cuda_rtf > 1.0 else 'No'}")
        
        print(f"\nComparison:")
        print(f"  CUDA Speedup: {speedup:.2f}x")
        print(f"  RTF Improvement: {rtf_improvement:.2f}x")
        print(f"  Winner: {'CUDA' if speedup > 1.0 else 'CPU'}")
        
        if speedup > 1.0:
            print(f"  CUDA is {speedup:.1f}x faster than CPU")
        else:
            print(f"  CPU is {1/speedup:.1f}x faster than CUDA (unexpected)")
        
        # Performance assertions
        assert cpu_avg_time > 0, "CPU processing time should be positive"
        assert cuda_avg_time > 0, "CUDA processing time should be positive"
        
        # Usually CUDA should be faster, but not always guaranteed
        if speedup < 0.5:  # If CPU is more than 2x faster than CUDA, something might be wrong
            print(f"WARNING: CPU appears significantly faster than CUDA (speedup: {speedup:.2f}x)")
