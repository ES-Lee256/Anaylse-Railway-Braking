"""Bearing-fault CNN training and inference on measured vibration signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from condition_cnn import Trainable1DCNN, TrainingSummary

BEARING_CLASS_NAMES = (
    "健康轴承",
    "内圈故障",
    "外圈故障",
    "滚动体故障",
    "内外圈及滚动体复合故障",
)
FILE_PREFIX_LABELS = {"H": 0, "I": 1, "O": 2, "B": 3, "C": 4}
WINDOW_SIZE = 512
WINDOW_STEP = 2048
DECIMATION_FACTOR = 10
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bearing_condition_v2.npz"


@dataclass(frozen=True)
class BearingTrainingResult:
    summary: TrainingSummary
    training_sample_count: int
    validation_sample_count: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    precision_by_class: tuple[float, ...]
    recall_by_class: tuple[float, ...]


def _channel_one(mat_data: dict[str, object], path: Path) -> np.ndarray:
    for key in ("Channel_1", "channel_1", "Channel1", "channel1"):
        if key in mat_data:
            channel = np.asarray(mat_data[key], dtype=np.float64).squeeze()
            if channel.ndim == 1 and channel.size >= WINDOW_SIZE * DECIMATION_FACTOR:
                return channel
    available = ", ".join(sorted(key for key in mat_data if not key.startswith("__")))
    raise ValueError(
        f"{path.name} 中未找到一维加速度 Channel_1；可用字段：{available}"
    )


def bearing_label_from_filename(path: Path) -> int:
    prefix = path.stem.split("-", maxsplit=1)[0].upper()
    if prefix not in FILE_PREFIX_LABELS:
        raise ValueError(f"无法从文件名识别轴承类别：{path.name}")
    return FILE_PREFIX_LABELS[prefix]


def load_bearing_windows(path: Path) -> np.ndarray:
    """Load accelerometer data and return overlapping 2-channel CNN windows."""
    try:
        raw = _channel_one(loadmat(path), path)
    except NotImplementedError as error:
        raise ValueError(
            f"{path.name} 是不受当前 SciPy MAT 读取器支持的格式。"
        ) from error
    if not np.isfinite(raw).all():
        raise ValueError(f"{path.name} 的加速度信号含 NaN 或无穷值。")

    signal = resample_poly(raw, up=1, down=DECIMATION_FACTOR)
    if signal.size < WINDOW_SIZE:
        raise ValueError(f"{path.name} 的有效信号短于一个 CNN 窗口。")
    starts = np.arange(0, signal.size - WINDOW_SIZE + 1, WINDOW_STEP)
    windows = np.empty((starts.size, WINDOW_SIZE, 2), dtype=np.float64)
    for index, start in enumerate(starts):
        segment = signal[start : start + WINDOW_SIZE]
        centered = segment - segment.mean()
        rms = float(np.sqrt(np.mean(centered**2)))
        if not np.isfinite(rms):
            raise ValueError(f"{path.name} 的振动窗口无法转换为有限值。")
        windows[index, :, 0] = centered / max(rms, 1e-12)
        windows[index, :, 1] = np.log10(max(rms, 1e-12))
    return windows


def _predict_batched(model: Trainable1DCNN, signals: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            model.predict_proba(signals[start : start + 64])
            for start in range(0, signals.shape[0], 64)
        ],
        axis=0,
    )


def train_bearing_model(
    data_directory: Path,
    model_path: Path = MODEL_PATH,
    *,
    epochs: int = 80,
    seed: int = 2026,
) -> BearingTrainingResult:
    """Train on trials 01/02 and evaluate on untouched trial 03 per class."""
    training_signals: list[np.ndarray] = []
    training_labels: list[np.ndarray] = []
    validation_signals: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    required_files = {
        f"{prefix}-A-{trial}.mat"
        for prefix in FILE_PREFIX_LABELS
        for trial in range(1, 4)
    }
    missing = sorted(name for name in required_files if not (data_directory / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"训练数据不完整，请先运行 download_bearing_dataset.py：{', '.join(missing)}"
        )

    for prefix, label in FILE_PREFIX_LABELS.items():
        for trial in (1, 2, 3):
            path = data_directory / f"{prefix}-A-{trial}.mat"
            windows = load_bearing_windows(path)
            labels = np.full(windows.shape[0], label, dtype=np.int64)
            if trial == 3:
                validation_signals.append(windows)
                validation_labels.append(labels)
            else:
                training_signals.append(windows)
                training_labels.append(labels)

    signals = np.concatenate(training_signals + validation_signals, axis=0)
    labels = np.concatenate(training_labels + validation_labels)
    validation_indices = np.arange(
        sum(array.shape[0] for array in training_signals),
        signals.shape[0],
        dtype=np.int64,
    )
    model = Trainable1DCNN(BEARING_CLASS_NAMES, input_channels=2, seed=seed)
    training_count = int(validation_indices[0])
    input_mean = signals[:training_count].mean(axis=(0, 1))
    input_scale = signals[:training_count].std(axis=(0, 1))
    input_scale[input_scale < 1e-12] = 1.0
    model.set_input_normalizer(input_mean, input_scale)
    summary = model.fit(
        signals,
        labels,
        epochs=epochs,
        batch_size=64,
        learning_rate=0.035,
        seed=seed + 1,
        validation_indices=validation_indices,
    )

    validation_probabilities = _predict_batched(
        model, signals[validation_indices]
    )
    predictions = validation_probabilities.argmax(axis=1)
    truth = labels[validation_indices]
    confusion = np.zeros(
        (len(BEARING_CLASS_NAMES), len(BEARING_CLASS_NAMES)), dtype=np.int64
    )
    np.add.at(confusion, (truth, predictions), 1)
    precision = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(len(BEARING_CLASS_NAMES), dtype=np.float64),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(BEARING_CLASS_NAMES), dtype=np.float64),
        where=confusion.sum(axis=1) != 0,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return BearingTrainingResult(
        summary=summary,
        training_sample_count=training_count,
        validation_sample_count=int(len(validation_indices)),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        precision_by_class=tuple(float(value) for value in precision),
        recall_by_class=tuple(float(value) for value in recall),
    )


def predict_bearing_file(model: Trainable1DCNN, path: Path) -> tuple[int, np.ndarray]:
    windows = load_bearing_windows(path)
    probabilities = _predict_batched(model, windows)
    return int(probabilities.mean(axis=0).argmax()), probabilities.mean(axis=0)
