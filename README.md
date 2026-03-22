# Whiskerscope

Real-time cat detector using your webcam. Powered by YOLOv8 + OpenCV.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- Live webcam cat detection with bounding boxes
- Real-time counters (cats now, total detections, max simultaneous)
- Optional clip recording when cats are detected
- REST API for programmatic access

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Streamlit dashboard (webcam + live UI)
streamlit run src/whiskerscope/adapters/streamlit_app.py

# REST API
uvicorn whiskerscope.adapters.fastapi_app:app --reload

# Run tests
pytest
```

## API

```bash
# Detect cats in an image
curl -X POST http://localhost:8000/detect -F "file=@cat.jpg"

# Check session stats
curl http://localhost:8000/status
```

## Architecture

Clean Architecture with three layers:

```
domain/       Pure Python models and port interfaces (zero external deps)
application/  Use cases: detection service, clip recorder, event counter
adapters/     YOLOv8, OpenCV, Streamlit, FastAPI implementations
```

## Stack

Python 3.11+ | YOLOv8 (nano) | OpenCV | Streamlit | FastAPI
