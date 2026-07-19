from pathlib import Path
import platform

import torch
import ultralytics
from ultralytics import YOLO


DATA_YAML = Path("/home/data/yoloA27/CAUCAFall_YOLO/data.yaml")
MODEL_PATH = Path("/home/data/yoloA27/yolo26n.pt")
DATASET_ROOT = Path("/home/data/yoloA27/CAUCAFall_YOLO")
PROJECT_DIR = Path("/home/data/yoloA27/experiments")
RUN_NAME = "yolo26n_baseline_seed42"

EXPECTED_COUNTS = {
    "train": 11818,
    "val": 4070,
    "test": 3958,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset():
    """训练前检查数据集路径、数量和图片标签配对。"""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"找不到 data.yaml：{DATA_YAML}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型权重：{MODEL_PATH}")

    for split, expected_count in EXPECTED_COUNTS.items():
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"找不到图片目录：{image_dir}")

        if not label_dir.is_dir():
            raise FileNotFoundError(f"找不到标签目录：{label_dir}")

        image_files = [
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        label_files = [
            path for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ]

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        print(
            f"{split}: images={len(image_files)}, "
            f"labels={len(label_files)}, expected={expected_count}"
        )

        if len(image_files) != expected_count:
            raise RuntimeError(
                f"{split} 图片数量错误："
                f"实际 {len(image_files)}，预期 {expected_count}"
            )

        if len(label_files) != expected_count:
            raise RuntimeError(
                f"{split} 标签数量错误："
                f"实际 {len(label_files)}，预期 {expected_count}"
            )

        if missing_labels:
            raise RuntimeError(
                f"{split} 有 {len(missing_labels)} 张图片缺少标签"
            )

        if missing_images:
            raise RuntimeError(
                f"{split} 有 {len(missing_images)} 个标签缺少图片"
            )

    print("数据集检查通过。")


def print_environment():
    """打印并记录当前实验环境。"""
    print("\n实验环境")
    print(f"Python: {platform.python_version()}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")


def main():
    check_dataset()
    print_environment()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        seed=42,
        deterministic=True,
        cache=False,
        amp=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        save_period=10,
        plots=True,
        val=True,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()