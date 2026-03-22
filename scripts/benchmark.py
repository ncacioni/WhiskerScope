"""Benchmark YOLO vs ONNX detector performance."""

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from whiskerscope.adapters.yolo_detector import YOLOv8Detector
from whiskerscope.domain.models import Frame

N_FRAMES = 50


def make_dummy_frame() -> Frame:
    data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Frame(data=data, timestamp=datetime.now(timezone.utc), width=640, height=480)


def benchmark_detector(name: str, detector, frames: list[Frame]) -> dict:
    times = []
    for f in frames:
        t0 = time.perf_counter()
        detector.detect_cats(f)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    fps = 1000 / avg if avg > 0 else 0
    return {"name": name, "avg_ms": round(avg, 1), "fps": round(fps, 1), "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 1)}


def main():
    frames = [make_dummy_frame() for _ in range(N_FRAMES)]
    results = []

    print(f"Benchmarking with {N_FRAMES} frames...\n")

    # YOLO
    yolo = YOLOv8Detector("yolov8n.pt", confidence=0.5)
    results.append(benchmark_detector("YOLOv8 (nano)", yolo, frames))

    # ONNX (optional)
    try:
        from whiskerscope.adapters.onnx_detector import ONNXDetector
        onnx = ONNXDetector("yolov8n.onnx", confidence=0.5)
        results.append(benchmark_detector("ONNX Runtime", onnx, frames))
    except Exception as e:
        print(f"ONNX skipped: {e}\n")

    # Print results
    print(f"| {'Detector':<20} | {'Avg (ms)':>10} | {'FPS':>8} | {'P95 (ms)':>10} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*10}|{'-'*12}|")
    for r in results:
        print(f"| {r['name']:<20} | {r['avg_ms']:>10} | {r['fps']:>8} | {r['p95_ms']:>10} |")


if __name__ == "__main__":
    main()
