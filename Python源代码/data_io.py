"""CSV input and output helpers."""

import csv
from pathlib import Path
from typing import Iterable

from models import BrakingCase, BrakingResult

CASE_FIELDS = (
    "name",
    "mass_kg",
    "initial_speed_kmh",
    "braking_ratio",
    "response_time_s",
    "grade_permille",
    "basic_resistance_coefficient",
    "rotating_mass_factor",
    "adhesion_coefficient",
)

RESULT_FIELDS = (
    "name",
    "mass_kg",
    "initial_speed_kmh",
    "braking_ratio",
    "response_time_s",
    "grade_permille",
    "basic_resistance_coefficient",
    "rotating_mass_factor",
    "adhesion_coefficient",
    "deceleration_m_s2",
    "braking_force_n",
    "adhesion_limit_force_n",
    "response_distance_m",
    "braking_distance_m",
    "stopping_distance_m",
    "braking_time_s",
    "total_time_s",
)

_NUMERIC_FIELDS = CASE_FIELDS[1:]


def load_cases(path: str | Path) -> list[BrakingCase]:
    """Load and validate cases from a UTF-8 CSV file."""
    cases: list[BrakingCase] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 文件为空或缺少表头。")
        missing_fields = [field for field in CASE_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(
                "CSV 缺少必需列：" + ", ".join(missing_fields)
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError("存在多余的数据列。")
                if any(row.get(field) is None or not row[field].strip() for field in CASE_FIELDS):
                    raise ValueError("存在空字段。")
                values = {field: float(row[field]) for field in _NUMERIC_FIELDS}
                cases.append(BrakingCase(name=row["name"].strip(), **values))
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV 第 {row_number} 行无效：{error}") from error

    if not cases:
        raise ValueError("CSV 中没有可导入的工况数据。")
    return cases


def export_results(path: str | Path, results: Iterable[BrakingResult]) -> None:
    """Write successful calculation results to a UTF-8 CSV file."""
    result_list = list(results)
    if not result_list:
        raise ValueError("没有可导出的计算结果。")

    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in result_list:
            writer.writerow(
                {
                    "name": result.case.name,
                    "mass_kg": result.case.mass_kg,
                    "initial_speed_kmh": result.case.initial_speed_kmh,
                    "braking_ratio": result.case.braking_ratio,
                    "response_time_s": result.case.response_time_s,
                    "grade_permille": result.case.grade_permille,
                    "basic_resistance_coefficient": (
                        result.case.basic_resistance_coefficient
                    ),
                    "rotating_mass_factor": result.case.rotating_mass_factor,
                    "adhesion_coefficient": result.case.adhesion_coefficient,
                    "deceleration_m_s2": result.deceleration_m_s2,
                    "braking_force_n": result.braking_force_n,
                    "adhesion_limit_force_n": result.adhesion_limit_force_n,
                    "response_distance_m": result.response_distance_m,
                    "braking_distance_m": result.braking_distance_m,
                    "stopping_distance_m": result.stopping_distance_m,
                    "braking_time_s": result.braking_time_s,
                    "total_time_s": result.total_time_s,
                }
            )
