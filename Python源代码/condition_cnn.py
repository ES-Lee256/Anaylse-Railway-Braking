"""A small 1D CNN trained only on synthetic railway condition signals."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from models import BrakingCase
from runaway import assess_runaway

CLASS_NAMES = (
    "合成常规工况",
    "合成部件载荷升高",
    "合成耐受极限超出",
    "合成极端整体失效类别（非真实散架预测）",
)
SYNTHETIC_TRAINING_NOTICE = (
    "训练集、标签阈值和耐受等级均为程序生成的教学演示数据，"
    "没有使用真实车辆试验、有限元或部件材料数据。"
)
_CHANNEL_SCALES = np.array([3.0, 9.0, 6.0, 4.0, 20.0], dtype=np.float64)
_CLASS_THRESHOLDS = (0.75, 1.35, 2.5)
_KERNEL_SIZE = 5
_FILTERS = 12
_EPOCHS = 40
_BATCH_SIZE = 64


@dataclass(frozen=True)
class TrainingSummary:
    validation_accuracy: float
    sample_count: int
    epochs: int


@dataclass(frozen=True)
class Prediction:
    class_index: int
    class_name: str
    confidence: float
    probabilities: tuple[float, ...]
    synthetic_load_index: float
    input_signal: np.ndarray


class Trainable1DCNN:
    """Trainable 1D convolution + global pooling + softmax classifier."""

    def __init__(
        self,
        class_names: tuple[str, ...],
        input_channels: int,
        seed: int = 0,
    ) -> None:
        if len(class_names) < 2 or input_channels < 1:
            raise ValueError("CNN 至少需要两个类别和一个输入通道。")
        self.class_names = tuple(class_names)
        rng = np.random.default_rng(seed)
        self.conv_weights = rng.normal(
            0.0,
            math.sqrt(2 / (_KERNEL_SIZE * input_channels)),
            size=(_KERNEL_SIZE, input_channels, _FILTERS),
        )
        self.conv_bias = np.zeros(_FILTERS, dtype=np.float64)
        self.dense_weights = rng.normal(
            0.0, math.sqrt(2 / _FILTERS), size=(_FILTERS, len(self.class_names))
        )
        self.dense_bias = np.zeros(len(self.class_names), dtype=np.float64)
        self.input_mean = np.zeros(input_channels, dtype=np.float64)
        self.input_scale = np.ones(input_channels, dtype=np.float64)
        self.summary: TrainingSummary | None = None

    def set_input_normalizer(
        self, mean: np.ndarray, scale: np.ndarray
    ) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if (
            mean.shape != self.input_mean.shape
            or scale.shape != self.input_scale.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("输入归一化参数必须是有限且为正的通道向量。")
        self.input_mean = mean.copy()
        self.input_scale = scale.copy()

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            class_names=np.asarray(self.class_names, dtype=np.str_),
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            conv_weights=self.conv_weights,
            conv_bias=self.conv_bias,
            dense_weights=self.dense_weights,
            dense_bias=self.dense_bias,
        )

    @classmethod
    def load(cls, path: Path) -> "Trainable1DCNN":
        with np.load(path, allow_pickle=False) as archive:
            class_names = tuple(str(value) for value in archive["class_names"])
            input_mean = archive["input_mean"].copy()
            input_scale = archive["input_scale"].copy()
            conv_weights = archive["conv_weights"].copy()
            conv_bias = archive["conv_bias"].copy()
            dense_weights = archive["dense_weights"].copy()
            dense_bias = archive["dense_bias"].copy()
        if (
            conv_weights.ndim != 3
            or conv_weights.shape[0] != _KERNEL_SIZE
            or conv_bias.shape != (conv_weights.shape[2],)
            or dense_weights.shape != (conv_weights.shape[2], len(class_names))
            or dense_bias.shape != (len(class_names),)
        ):
            raise ValueError("已保存的 CNN 权重文件结构无效。")
        model = cls(class_names, conv_weights.shape[1])
        model.set_input_normalizer(input_mean, input_scale)
        model.conv_weights = conv_weights
        model.conv_bias = conv_bias
        model.dense_weights = dense_weights
        model.dense_bias = dense_bias
        return model

    def _forward(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output_steps = signals.shape[1] - _KERNEL_SIZE + 1
        convolution = np.zeros(
            (signals.shape[0], output_steps, _FILTERS), dtype=np.float64
        )
        for offset in range(_KERNEL_SIZE):
            window = signals[:, offset : offset + output_steps, :]
            convolution += np.einsum(
                "ntc,cf->ntf",
                window,
                self.conv_weights[offset],
                optimize=True,
            )
        convolution += self.conv_bias
        activated = np.maximum(convolution, 0)
        pooled = activated.mean(axis=1)
        logits = pooled @ self.dense_weights + self.dense_bias
        return convolution, pooled, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = _EPOCHS,
        batch_size: int = _BATCH_SIZE,
        learning_rate: float = 0.035,
        seed: int = 0,
        validation_indices: np.ndarray | None = None,
    ) -> TrainingSummary:
        if signals.ndim != 3 or signals.shape[1] < _KERNEL_SIZE:
            raise ValueError("训练信号必须是 (样本, 时间步, 通道) 三维数组。")
        if signals.shape[2] != self.conv_weights.shape[1]:
            raise ValueError("训练信号通道数与卷积模型不一致。")
        if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
            raise ValueError("每条训练信号都必须对应一个类别标签。")
        if signals.shape[0] < len(self.class_names) * 2:
            raise ValueError("训练样本不足，无法覆盖所有分类类别。")
        if (
            not np.isfinite(signals).all()
            or not np.isin(labels, np.arange(len(self.class_names))).all()
        ):
            raise ValueError("训练信号或标签包含无效值。")
        signals = (signals - self.input_mean) / self.input_scale
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("训练轮数、批量大小和学习率必须为正数。")

        rng = np.random.default_rng(seed)
        if validation_indices is None:
            indices = rng.permutation(signals.shape[0])
            validation_count = max(1, int(signals.shape[0] * 0.2))
            validation_indices = indices[:validation_count]
        else:
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            if (
                validation_indices.ndim != 1
                or validation_indices.size == 0
                or np.any(validation_indices < 0)
                or np.any(validation_indices >= signals.shape[0])
                or np.unique(validation_indices).size != validation_indices.size
            ):
                raise ValueError("验证集索引必须是非空且不重复的有效索引。")
        validation_mask = np.zeros(signals.shape[0], dtype=bool)
        validation_mask[validation_indices] = True
        training_indices = np.flatnonzero(~validation_mask)
        if (
            training_indices.size == 0
            or np.unique(labels[training_indices]).size != len(self.class_names)
        ):
            raise ValueError("训练拆分未覆盖全部类别或没有训练样本。")
        if np.unique(labels[validation_indices]).size != len(self.class_names):
            raise ValueError("验证拆分未覆盖全部类别。")

        x_train = signals[training_indices]
        y_train = labels[training_indices]
        for _ in range(epochs):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, order.size, batch_size):
                batch_indices = order[start : start + batch_size]
                batch = x_train[batch_indices]
                targets = y_train[batch_indices]
                convolution, pooled, logits = self._forward(batch)
                probabilities = self._softmax(logits)
                probabilities[np.arange(targets.size), targets] -= 1
                probabilities /= targets.size

                dense_gradient = pooled.T @ probabilities
                dense_bias_gradient = probabilities.sum(axis=0)
                pooled_gradient = probabilities @ self.dense_weights.T
                convolution_gradient = (
                    np.repeat(
                        pooled_gradient[:, np.newaxis, :],
                        convolution.shape[1],
                        axis=1,
                    )
                    / convolution.shape[1]
                )
                convolution_gradient *= convolution > 0

                conv_weight_gradient = np.zeros_like(self.conv_weights)
                conv_bias_gradient = convolution_gradient.sum(axis=(0, 1))
                for offset in range(_KERNEL_SIZE):
                    window = batch[
                        :, offset : offset + convolution.shape[1], :
                    ]
                    conv_weight_gradient[offset] = np.einsum(
                        "ntc,ntf->cf",
                        window,
                        convolution_gradient,
                        optimize=True,
                    )

                self.dense_weights -= learning_rate * dense_gradient
                self.dense_bias -= learning_rate * dense_bias_gradient
                self.conv_weights -= learning_rate * conv_weight_gradient
                self.conv_bias -= learning_rate * conv_bias_gradient

        _, _, validation_logits = self._forward(signals[validation_indices])
        validation_probabilities = self._softmax(validation_logits)
        accuracy = float(
            (
                validation_probabilities.argmax(axis=1)
                == labels[validation_indices]
            ).mean()
        )
        self.summary = TrainingSummary(
            validation_accuracy=accuracy,
            sample_count=int(signals.shape[0]),
            epochs=epochs,
        )
        return self.summary

    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        if signals.ndim != 3 or signals.shape[2] != self.conv_weights.shape[1]:
            raise ValueError("待预测信号的形状与模型输入不匹配。")
        if signals.shape[1] < _KERNEL_SIZE or not np.isfinite(signals).all():
            raise ValueError("待预测信号长度不足或包含无效数值。")
        signals = (signals - self.input_mean) / self.input_scale
        _, _, logits = self._forward(signals)
        return self._softmax(logits)


class SyntheticConditionCNN(Trainable1DCNN):
    """Classifier for fictional railway-condition demonstration data."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(CLASS_NAMES, len(_CHANNEL_SCALES) + 1, seed)

    def predict(self, case: BrakingCase) -> Prediction:
        if self.summary is None:
            raise ValueError("模型尚未完成训练。")
        signal, synthetic_load_index = case_to_signal(case)
        probabilities = self.predict_proba(signal[np.newaxis, ...])[0]
        class_index = int(probabilities.argmax())
        return Prediction(
            class_index=class_index,
            class_name=self.class_names[class_index],
            confidence=float(probabilities[class_index]),
            probabilities=tuple(float(value) for value in probabilities),
            synthetic_load_index=synthetic_load_index,
            input_signal=signal,
        )


def _features_for_case(
    case: BrakingCase, variation: float = 0.0
) -> tuple[np.ndarray, float]:
    assessment = assess_runaway(case)
    speed_factor = case.initial_speed_kmh / 80.0
    mass_factor = case.mass_kg / 40000.0
    braking_factor = case.braking_ratio / 0.12
    grade_factor = abs(case.grade_permille) / 30.0
    adhesion_factor = case.braking_ratio / case.adhesion_coefficient
    load_index = (
        0.16 * mass_factor
        + 0.28 * speed_factor**2
        + 0.18 * braking_factor
        + 0.12 * grade_factor
        + 0.26 * adhesion_factor
    )

    effective_duration = max(assessment.simulation_duration_s, 1e-6)
    time_fraction = np.linspace(0.0, 1.0, 40)
    signal = np.empty((time_fraction.size, 6), dtype=np.float64)
    for index, fraction in enumerate(time_fraction):
        _, speed_m_s = assessment.state_at(
            fraction * assessment.simulation_duration_s
        )
        pressure_build = min(
            1.0,
            max(
                fraction * effective_duration
                / max(case.response_time_s, 0.5),
                0.0,
            ),
        )
        values = np.array(
            [
                mass_factor / 3.0,
                speed_factor**2 / 9.0,
                braking_factor / 6.0,
                grade_factor / 4.0,
                adhesion_factor / 20.0,
                pressure_build * speed_m_s / max(speed_factor * 80 / 3.6, 1e-6),
            ],
            dtype=np.float64,
        )
        values += variation * np.array(
            [
                0.25,
                0.20,
                0.15,
                0.20,
                0.10,
                0.15,
            ]
        ) * np.sin(2 * math.pi * fraction + index * 0.08)
        signal[index] = values
    return signal, load_index


def case_to_signal(case: BrakingCase) -> tuple[np.ndarray, float]:
    """Build the synthetic time-series input for a user-selected case."""
    try:
        signal, load_index = _features_for_case(case)
    except OverflowError as error:
        raise ValueError("当前工况无法转换为有限的模型输入。") from error
    if not np.isfinite(signal).all() or not math.isfinite(load_index):
        raise ValueError("当前工况无法转换为有限的模型输入。")
    return signal, load_index


def generate_synthetic_dataset(
    sample_count: int = 1600, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fictional signals and labels; not measured vehicle data."""
    if sample_count < len(CLASS_NAMES) * 8:
        raise ValueError("合成样本数至少为类别数的 8 倍。")
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    target_per_class = sample_count // len(CLASS_NAMES)
    class_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    attempts = 0
    max_attempts = sample_count * 100

    while np.any(class_counts < target_per_class) and attempts < max_attempts:
        attempts += 1
        case = BrakingCase(
            name=f"合成样本-{attempts}",
            mass_kg=float(rng.uniform(10000, 120000)),
            initial_speed_kmh=float(rng.uniform(0, 300)),
            braking_ratio=float(rng.uniform(0.02, 0.9)),
            response_time_s=float(rng.uniform(0.1, 5.0)),
            grade_permille=float(rng.uniform(-50, 50)),
            basic_resistance_coefficient=float(rng.uniform(0.0005, 0.01)),
            rotating_mass_factor=float(rng.uniform(0.02, 0.2)),
            adhesion_coefficient=float(rng.uniform(0.03, 0.8)),
        )
        load_index = _features_for_case(case, variation=0.01)[1]
        label = sum(load_index >= threshold for threshold in _CLASS_THRESHOLDS)
        if class_counts[label] >= target_per_class:
            continue
        signal, _ = _features_for_case(case, variation=0.02)
        signals.append(signal)
        labels.append(label)
        class_counts[label] += 1

    if np.any(class_counts < target_per_class):
        raise ValueError("未能生成覆盖所有类别的合成训练样本。")

    signal_array = np.stack(signals)
    label_array = np.asarray(labels, dtype=np.int64)
    return signal_array, label_array


def train_synthetic_model(
    sample_count: int = 2400, seed: int = 0
) -> tuple[SyntheticConditionCNN, TrainingSummary]:
    """Generate fictional training data and fit a fresh CNN model."""
    signals, labels = generate_synthetic_dataset(
        sample_count=sample_count, seed=seed
    )
    model = SyntheticConditionCNN(seed=seed)
    summary = model.fit(
        signals,
        labels,
        epochs=_EPOCHS,
        batch_size=_BATCH_SIZE,
        seed=seed + 1,
    )
    return model, summary
