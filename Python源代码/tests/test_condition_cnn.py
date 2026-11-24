import unittest

import numpy as np

from condition_cnn import (
    CLASS_NAMES,
    SyntheticConditionCNN,
    case_to_signal,
    generate_synthetic_dataset,
)
from models import BrakingCase


def make_case(**overrides: float | str) -> BrakingCase:
    values: dict[str, float | str] = {
        "name": "测试工况",
        "mass_kg": 40000.0,
        "initial_speed_kmh": 80.0,
        "braking_ratio": 0.12,
        "response_time_s": 1.0,
        "grade_permille": 0.0,
        "basic_resistance_coefficient": 0.002,
        "rotating_mass_factor": 0.06,
        "adhesion_coefficient": 0.25,
    }
    values.update(overrides)
    return BrakingCase(**values)  # type: ignore[arg-type]


class SyntheticConditionCnnTests(unittest.TestCase):
    def test_dataset_has_balanced_labels_and_time_series(self) -> None:
        signals, labels = generate_synthetic_dataset(320, seed=4)
        self.assertEqual(signals.shape, (320, 40, 6))
        self.assertEqual(labels.shape, (320,))
        self.assertEqual(np.unique(labels).size, len(CLASS_NAMES))
        self.assertTrue(np.isfinite(signals).all())

    def test_model_trains_and_predicts_extreme_synthetic_class(self) -> None:
        signals, labels = generate_synthetic_dataset(2400, seed=7)
        model = SyntheticConditionCNN(seed=7)
        summary = model.fit(
            signals,
            labels,
            epochs=40,
            batch_size=64,
            seed=8,
        )
        self.assertGreaterEqual(summary.validation_accuracy, 0.75)
        self.assertEqual(summary.sample_count, 2400)

        extreme_case = make_case(
            mass_kg=120000.0,
            initial_speed_kmh=300.0,
            braking_ratio=0.9,
            grade_permille=50.0,
            adhesion_coefficient=0.03,
        )
        prediction = model.predict(extreme_case)
        self.assertEqual(
            prediction.class_name,
            "合成极端整体失效类别（非真实散架预测）",
        )
        self.assertGreater(prediction.synthetic_load_index, 2.5)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)

    def test_prediction_requires_a_trained_model(self) -> None:
        model = SyntheticConditionCNN()
        with self.assertRaisesRegex(ValueError, "尚未完成训练"):
            model.predict(make_case())

    def test_extreme_input_overflow_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            case_to_signal(make_case(initial_speed_kmh=1e308))

    def test_invalid_signal_shape_is_rejected(self) -> None:
        signals = np.zeros((8, 40, 6), dtype=np.float64)
        labels = np.zeros(8, dtype=np.int64)
        model = SyntheticConditionCNN()
        with self.assertRaises(ValueError):
            model.fit(signals, labels)


if __name__ == "__main__":
    unittest.main()
