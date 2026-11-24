import csv
import tempfile
import unittest
from pathlib import Path

from calculations import calculate_braking
from data_io import CASE_FIELDS, export_results, load_cases
from models import BrakingCase


class CsvDataTests(unittest.TestCase):
    def test_sample_csv_loads_all_cases(self) -> None:
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "braking_cases.csv"
        cases = load_cases(sample_path)
        self.assertEqual(len(cases), 13)
        self.assertEqual(cases[0].name, "平直线路")
        for case in cases:
            with self.subTest(case=case.name):
                self.assertGreater(calculate_braking(case).stopping_distance_m, 0)

    def test_vehicle_type_examples_all_load_and_calculate(self) -> None:
        examples_directory = Path(__file__).resolve().parents[1] / "examples"
        vehicle_files = sorted(examples_directory.glob("车型_*.csv"))
        self.assertEqual(len(vehicle_files), 10)

        for path in vehicle_files:
            with self.subTest(file=path.name):
                cases = load_cases(path)
                self.assertEqual(len(cases), 3)
                for case in cases:
                    self.assertGreater(
                        calculate_braking(case).stopping_distance_m, 0
                    )

    def test_csv_errors_identify_the_invalid_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(CASE_FIELDS)
                writer.writerow(
                    ("错误工况", "40000", "非数字", "0.12", "1", "0", "0.002", "0.06", "0.25")
                )
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                load_cases(path)

    def test_export_contains_calculated_results(self) -> None:
        case = BrakingCase(
            name="导出测试",
            mass_kg=40000,
            initial_speed_kmh=80,
            braking_ratio=0.12,
            response_time_s=1,
            grade_permille=0,
            basic_resistance_coefficient=0.002,
            rotating_mass_factor=0.06,
            adhesion_coefficient=0.25,
        )
        result = calculate_braking(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            export_results(path, [result])
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "导出测试")
            self.assertEqual(float(rows[0]["response_time_s"]), 1.0)
            self.assertEqual(float(rows[0]["adhesion_coefficient"]), 0.25)
            self.assertAlmostEqual(
                float(rows[0]["stopping_distance_m"]),
                result.stopping_distance_m,
            )


if __name__ == "__main__":
    unittest.main()
