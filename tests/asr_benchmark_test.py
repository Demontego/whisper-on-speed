import datetime
import time
from pathlib import Path

import pytest
import torch

from src.core.pipe import Pipe
from tests.test_utils import (
    calculate_performance_stats,
    calculate_rtf_metrics,
    get_gpu_info,
    measure_batch_transcription_time,
    measure_transcription_time,
    print_gpu_info,
    print_performance_summary,
    save_benchmark_results,
)


class TestASRBenchmark:
    """Benchmark tests for ASR"""

    @pytest.mark.benchmark
    def test_single_transcription_performance(
        self, pipe_warmed: Pipe, test_audio_files: list[dict], benchmark_results_dir: Path
    ) -> None:
        """Benchmark test for single file transcription performance"""
        gpu_info = get_gpu_info()
        audio_duration = len(test_audio_files[0]['audio']) / 16000
        num_runs = 5

        # Measure performance
        times = measure_transcription_time(pipe_warmed, test_audio_files[0]['audio'], num_runs)

        # Calculate statistics
        stats = calculate_performance_stats(times)
        rtf_metrics = calculate_rtf_metrics(audio_duration, times)

        # Prepare results data
        results_data = {
            'test_name': 'single_transcription_performance',
            'timestamp': datetime.datetime.now().isoformat(),
            'model_id': pipe_warmed.config_asr.model_id,
            'num_runs': num_runs,
            'execution_times': times,
            'statistics': stats,
            'rtf_metrics': rtf_metrics,
            'audio_info': {'sample_rate': 16000, 'duration_seconds': audio_duration},
            'hardware_info': {'gpu': gpu_info},
        }

        # Save and print results
        save_benchmark_results(benchmark_results_dir, 'single_transcription', results_data)
        print_performance_summary(
            'Single Transcription Benchmark', gpu_info, audio_duration, num_runs, stats, rtf_metrics
        )

        # Performance assertion
        assert stats['average_time'] < 30.0, f'Transcription too slow: {stats["average_time"]:.2f}s'

    @pytest.mark.benchmark
    def test_batch_transcription_performance(
        self, pipe_warmed: Pipe, test_audio_files: list[dict], benchmark_results_dir: Path
    ) -> None:
        """Benchmark test for batch transcription performance"""

        cnt = 40
        gpu_info = get_gpu_info()
        total_audio_duration = len(test_audio_files[0]['audio']) / 16000 * cnt
        num_runs = 3

        # Measure performance
        times = measure_batch_transcription_time(
            pipe_warmed, [file['audio'] for file in test_audio_files] * cnt, num_runs
        )

        # Calculate statistics
        stats = calculate_performance_stats(times)
        rtf_metrics = calculate_rtf_metrics(total_audio_duration, times)

        # Additional batch-specific metrics
        avg_time_per_file = stats['average_time'] / len(test_audio_files)
        avg_rtf_per_file = (total_audio_duration / len(test_audio_files)) / avg_time_per_file

        # Prepare results data
        results_data = {
            'test_name': 'batch_transcription_performance',
            'timestamp': datetime.datetime.now().isoformat(),
            'model_id': pipe_warmed.config_asr.model_id,
            'num_runs': num_runs,
            'num_files': len(test_audio_files),
            'execution_times': times,
            'statistics': {**stats, 'average_time_per_file': avg_time_per_file},
            'rtf_metrics': {**rtf_metrics, 'average_rtf_per_file': avg_rtf_per_file},
            'audio_info': {
                'sample_rate': 16000,
                'total_files': len(test_audio_files),
                'total_audio_duration': total_audio_duration,
                'file_durations': [len(file['audio']) / 16000 for file in test_audio_files],
            },
            'hardware_info': {'gpu': gpu_info},
        }

        # Save results
        save_benchmark_results(benchmark_results_dir, 'batch_transcription', results_data)

        # Print results
        print_performance_summary(
            'Batch Transcription Benchmark', gpu_info, total_audio_duration, num_runs, stats, rtf_metrics
        )
        print(f'Time per file: {avg_time_per_file:.3f}s')
        print(f'RTF per file: {avg_rtf_per_file:.2f}x')

    @pytest.mark.benchmark
    def test_warmup_vs_cold_performance(
        self, pipe: Pipe, pipe_warmed: Pipe, test_audio_files: list[dict], benchmark_results_dir: Path
    ) -> None:
        """Benchmark comparison of performance with and without warmup"""
        gpu_info = get_gpu_info()
        audio_duration = len(test_audio_files[0]['audio']) / 16000

        # Test without warmup
        start_time = time.time()
        cold_result = pipe([test_audio_files[0]['audio']])[0]
        cold_time = time.time() - start_time

        # Test with warmup
        start_time = time.time()
        warm_result = pipe_warmed([test_audio_files[0]['audio']])[0]
        warm_time = time.time() - start_time

        # Validate results
        assert isinstance(cold_result, list) and len(cold_result) > 0
        assert isinstance(warm_result, list) and len(warm_result) > 0
        assert cold_result[0].text is not None
        assert warm_result[0].text is not None

        # Calculate metrics
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        cold_rtf = audio_duration / cold_time
        warm_rtf = audio_duration / warm_time
        rtf_improvement = warm_rtf / cold_rtf if cold_rtf > 0 else float('inf')

        # Prepare results data
        results_data = {
            'test_name': 'warmup_vs_cold_performance',
            'timestamp': datetime.datetime.now().isoformat(),
            'model_id': pipe.config_asr.model_id,
            'results': {
                'cold_start_time': cold_time,
                'warmed_up_time': warm_time,
                'speedup': speedup,
                'warmup_steps': 2,
            },
            'rtf_metrics': {
                'cold_rtf': cold_rtf,
                'warm_rtf': warm_rtf,
                'rtf_improvement': rtf_improvement,
                'cold_faster_than_realtime': cold_rtf > 1.0,
                'warm_faster_than_realtime': warm_rtf > 1.0,
            },
            'audio_info': {'sample_rate': 16000, 'duration_seconds': audio_duration},
            'hardware_info': {'gpu': gpu_info},
        }

        # Save results
        save_benchmark_results(benchmark_results_dir, 'warmup_comparison', results_data)

        # Print results
        print('\n=== Warmup vs Cold Performance Comparison ===')

        print_gpu_info(gpu_info)
        print(f'Audio duration: {audio_duration:.2f}s')
        print(f'Cold start time: {cold_time:.3f}s')
        print(f'Warmed up time: {warm_time:.3f}s')
        print(f'Speedup: {speedup:.2f}x')
        print(f'Cold RTF: {cold_rtf:.2f}x')
        print(f'Warm RTF: {warm_rtf:.2f}x')
        print(f'RTF improvement: {rtf_improvement:.2f}x')
        print(f'Cold processing {"faster" if cold_rtf > 1.0 else "slower"} than real-time')
        print(f'Warm processing {"faster" if warm_rtf > 1.0 else "slower"} than real-time')

    @pytest.mark.benchmark
    def test_cpu_vs_cuda_performance(
        self, cpu_pipe: Pipe, pipe_warmed: Pipe, test_audio_files: list[dict], benchmark_results_dir: Path
    ) -> None:
        """Benchmark comparison of CPU vs CUDA performance"""
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available, skipping CPU vs CUDA comparison')

        gpu_info = get_gpu_info()
        audio_duration = len(test_audio_files[0]['audio']) / 16000
        num_runs = 3

        print('\n=== CPU vs CUDA Performance Comparison ===')
        print(f'Audio duration: {audio_duration:.2f}s')

        # Test with CPU
        print('Testing CPU performance...')
        cpu_times = measure_transcription_time(cpu_pipe, test_audio_files[0]['audio'], num_runs)
        cpu_stats = calculate_performance_stats(cpu_times)
        cpu_rtf = audio_duration / cpu_stats['average_time']

        # Test with CUDA
        print('Testing CUDA performance...')
        cuda_times = measure_transcription_time(pipe_warmed, test_audio_files[0]['audio'], num_runs)
        cuda_stats = calculate_performance_stats(cuda_times)
        cuda_rtf = audio_duration / cuda_stats['average_time']

        # Calculate comparison metrics
        speedup = (
            cpu_stats['average_time'] / cuda_stats['average_time'] if cuda_stats['average_time'] > 0 else float('inf')
        )
        rtf_improvement = cuda_rtf / cpu_rtf if cpu_rtf > 0 else float('inf')

        # Prepare results data
        results_data = {
            'test_name': 'cpu_vs_cuda_performance',
            'timestamp': datetime.datetime.now().isoformat(),
            'model_id': cpu_pipe.config_asr.model_id,
            'num_runs': num_runs,
            'audio_info': {'sample_rate': 16000, 'duration_seconds': audio_duration},
            'cpu_results': {
                'execution_times': cpu_times,
                'statistics': {**cpu_stats, 'rtf': cpu_rtf},
                'device': 'cpu',
                'faster_than_realtime': cpu_rtf > 1.0,
            },
            'cuda_results': {
                'execution_times': cuda_times,
                'statistics': {**cuda_stats, 'rtf': cuda_rtf},
                'device': 'cuda',
                'faster_than_realtime': cuda_rtf > 1.0,
            },
            'comparison': {
                'cuda_speedup': speedup,
                'rtf_improvement': rtf_improvement,
                'winner': 'cuda' if cuda_stats['average_time'] < cpu_stats['average_time'] else 'cpu',
            },
            'hardware_info': {'gpu': gpu_info},
        }

        # Save results
        save_benchmark_results(benchmark_results_dir, 'cpu_vs_cuda', results_data)

        # Print detailed comparison
        print('\n=== Results Summary ===')
        print('CPU Performance:')
        print(f'  Average time: {cpu_stats["average_time"]:.3f}s')
        print(f'  RTF: {cpu_rtf:.2f}x')
        print(f'  Faster than real-time: {"Yes" if cpu_rtf > 1.0 else "No"}')

        print('\nCUDA Performance:')
        print(f'  GPU: {gpu_info["device_name"]}')
        if gpu_info['cuda_available'] and gpu_info['memory_info']:
            memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
            print(f'  GPU Memory: {memory_gb:.1f}GB')
        print(f'  Average time: {cuda_stats["average_time"]:.3f}s')
        print(f'  RTF: {cuda_rtf:.2f}x')
        print(f'  Faster than real-time: {"Yes" if cuda_rtf > 1.0 else "No"}')

        print('\nComparison:')
        print(f'  CUDA Speedup: {speedup:.2f}x')
        print(f'  RTF Improvement: {rtf_improvement:.2f}x')
        print(f'  Winner: {"CUDA" if speedup > 1.0 else "CPU"}')

        if speedup > 1.0:
            print(f'  CUDA is {speedup:.1f}x faster than CPU')
        else:
            print(f'  CPU is {1 / speedup:.1f}x faster than CUDA (unexpected)')

        # Performance assertions
        assert cpu_stats['average_time'] > 0, 'CPU processing time should be positive'
        assert cuda_stats['average_time'] > 0, 'CUDA processing time should be positive'

        # Warning if CPU is significantly faster than CUDA
        if speedup < 0.5:
            print(f'WARNING: CPU appears significantly faster than CUDA (speedup: {speedup:.2f}x)')
