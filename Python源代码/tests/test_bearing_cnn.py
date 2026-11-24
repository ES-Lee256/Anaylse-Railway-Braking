import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from bearing_cnn import (
    BEARING_CLASS_NAMES,
    bearing_label_from_filename,
    load_bearing_windows,
)
from condition_cnn import Trainable1DCNN


class BearingCnnTests(unittest.TestCase):
    def test_matlab_acceleration_is_resampled_and_windowed(self) -> None:
        samples = np.arange(6400, dtype=np.float64)
        vibration = np.sin(2 * np.pi * 0.02 * samples)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "H-A-1.mat"
            savemat(path, {"Channel_1": vibration})
            windows = load_bearing_windows(path)

        self.assertEqual(windows.shape, (1, 512, 2))
        self.assertTrue(np.isfinite(windows).all())
        self.assertAlmostEqual(float(windows[:, :, 0].mean()), 0.0, places=12)

    def test_filename_prefix_maps_to_five_documented_classes(self) -> None:
        expected = {"H": 0, "I": 1, "O": 2, "B": 3, "C": 4}
        for prefix, label in expected.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    bearing_label_from_filename(Path(f"{prefix}-A-3.mat")),
                    label,
                )
        self.assertEqual(len(BEARING_CLASS_NAMES), 5)
        with self.assertRaises(ValueError):
            bearing_label_from_filename(Path("unknown.mat"))

    def test_grouped_split_training_and_weight_round_trip(self) -> None:
        rng = np.random.default_rng(12)
        labels = np.repeat(np.arange(5), 4)
        signals = rng.normal(size=(20, 12, 2))
        validation_indices = np.array([3, 7, 11, 15, 19])
        model = Trainable1DCNN(BEARING_CLASS_NAMES, input_channels=2, seed=3)
        summary = model.fit(
            signals,
            labels,
            epochs=1,
            batch_size=8,
            seed=4,
            validation_indices=validation_indices,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bearing_model.npz"
            model.save(path)
            restored = Trainable1DCNN.load(path)

        np.testing.assert_allclose(
            model.predict_proba(signals[:2]),
            restored.predict_proba(signals[:2]),
        )
        self.assertEqual(summary.sample_count, 20)
        expected_accuracy = float(
            (
                model.predict_proba(signals[validation_indices]).argmax(axis=1)
                == labels[validation_indices]
            ).mean()
        )
        self.assertAlmostEqual(summary.validation_accuracy, expected_accuracy)
        self.assertEqual(restored.class_names, BEARING_CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
