import datetime
import json
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.core.asr import ASRonSPEED
from src.core.replica import Replica


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for benchmarking"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'device_capability': None,
        'memory_info': None,
        'driver_version': None,
        'torch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_info.update({
            'current_device': current_device,
            'device_name': torch.cuda.get_device_name(current_device),
            'device_capability': torch.cuda.get_device_capability(current_device),
            'memory_info': {
                'total_memory': torch.cuda.get_device_properties(current_device).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(current_device),
                'memory_reserved': torch.cuda.memory_reserved(current_device),
                'max_memory_allocated': torch.cuda.max_memory_allocated(current_device),
                'max_memory_reserved': torch.cuda.max_memory_reserved(current_device),
            },
        })

        # Try to get CUDA driver version
        try:
            gpu_info['driver_version'] = torch.version.cuda  # type: ignore
        except Exception:
            gpu_info['driver_version'] = 'unknown'

    return gpu_info


def calculate_performance_stats(times: List[float]) -> Dict[str, float]:
    """Calculate performance statistics from execution times"""
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {'average_time': avg_time, 'minimum_time': min_time, 'maximum_time': max_time, 'std_deviation': std_dev}


def calculate_rtf_metrics(audio_duration: float, times: List[float]) -> Dict[str, float]:
    """Calculate Real-Time Factor metrics"""
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    avg_rtf = audio_duration / avg_time
    min_rtf = audio_duration / max_time  # Best case (min processing time)
    max_rtf = audio_duration / min_time  # Worst case (max processing time)

    return {
        'average_rtf': avg_rtf,
        'minimum_rtf': min_rtf,
        'maximum_rtf': max_rtf,
        'avg_audio_seconds_per_processing_second': avg_rtf,
        'processing_faster_than_realtime': avg_rtf > 1.0,
    }


def measure_transcription_time(model: ASRonSPEED, audio: np.ndarray, num_runs: int = 1) -> List[float]:
    """Measure transcription time for multiple runs"""
    times = []

    for _ in range(num_runs):
        start_time = time.time()
        result = model.process_audio(audio)
        end_time = time.time()
        times.append(end_time - start_time)

        # Validate result
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], Replica)
        assert result[0].text is not None
        assert len(result[0].text) > 0

    return times


def measure_batch_transcription_time(
    model: ASRonSPEED, audio_files: List[np.ndarray], num_runs: int = 1
) -> List[float]:
    """Measure batch transcription time for multiple runs"""
    times = []

    for _ in range(num_runs):
        start_time = time.time()
        results = model.process_batch(audio_files)
        end_time = time.time()
        times.append(end_time - start_time)

        # Validate results
        assert isinstance(results, list)
        assert len(results) == len(audio_files)

        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], Replica)
            assert result[0].text is not None
            assert len(result[0].text) > 0

    return times


def save_benchmark_results(benchmark_results_dir, test_name: str, results_data: Dict[str, Any]) -> None:
    """Save benchmark results to JSON file"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{test_name}_{timestamp}.json'
    filepath = benchmark_results_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f'Benchmark results saved to: {filepath}')


def print_gpu_info(gpu_info: Dict[str, Any]) -> None:
    """Print GPU information in a formatted way"""
    print(f'GPU: {gpu_info["device_name"] if gpu_info["cuda_available"] else "CPU only"}')
    if gpu_info['cuda_available'] and gpu_info['memory_info']:
        memory_gb = gpu_info['memory_info']['total_memory'] / (1024**3)
        print(f'GPU Memory: {memory_gb:.1f}GB')


def print_performance_summary(
    test_name: str,
    gpu_info: Dict[str, Any],
    audio_duration: float,
    num_runs: int,
    stats: Dict[str, float],
    rtf_metrics: Dict[str, float],
) -> None:
    """Print performance summary in a formatted way"""
    print(f'\n=== {test_name} ===')
    print_gpu_info(gpu_info)
    print(f'Audio duration: {audio_duration:.2f}s')
    print(f'Number of runs: {num_runs}')
    print(f'Average time: {stats["average_time"]:.3f}s')
    print(f'Minimum time: {stats["minimum_time"]:.3f}s')
    print(f'Maximum time: {stats["maximum_time"]:.3f}s')
    print(f'Std deviation: {stats["std_deviation"]:.3f}s')
    print(f'Average RTF: {rtf_metrics["average_rtf"]:.2f}x (аудио секунд за секунду обработки)')
    print(f'RTF range: {rtf_metrics["minimum_rtf"]:.2f}x - {rtf_metrics["maximum_rtf"]:.2f}x')
    print(f'Processing {"faster" if rtf_metrics["processing_faster_than_realtime"] else "slower"} than real-time')


def validate_asr_results(results, expected_count=None, check_timestamps=True):
    """
    Универсальная функция для проверки результатов ASR

    Args:
        results: результат ASR (может быть list для одиночного результата или list of lists для batch)
        expected_count: ожидаемое количество результатов для batch (None для одиночного результата)
        check_timestamps: проверять ли временные метки
    """
    if expected_count is not None:
        # Batch результаты
        assert isinstance(results, list)
        assert len(results) == expected_count

        for result in results:
            _validate_single_result(result, check_timestamps)
    else:
        # Одиночный результат
        _validate_single_result(results, check_timestamps)


def _validate_single_result(result, check_timestamps=True):
    """Проверка одиночного результата ASR"""
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], Replica)
    assert result[0].text is not None
    assert len(result[0].text) > 0

    if check_timestamps and result[0].start_time is not None and result[0].end_time is not None:
        assert result[0].start_time >= 0
        assert result[0].end_time > result[0].start_time
