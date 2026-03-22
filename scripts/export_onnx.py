"""Export YOLOv8 model to ONNX format."""

import sys
from ultralytics import YOLO


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    model = YOLO(model_name)
    path = model.export(format="onnx", simplify=True)
    print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
