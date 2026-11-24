"""Data models for the railway vehicle braking analysis application."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BrakingCase:
    name: str
    mass_kg: float
    initial_speed_kmh: float
    braking_ratio: float
    response_time_s: float
    grade_permille: float
    basic_resistance_coefficient: float
    rotating_mass_factor: float
    adhesion_coefficient: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("工况名称不能为空。")

        values = {
            "车辆质量": self.mass_kg,
            "初速度": self.initial_speed_kmh,
            "制动率": self.braking_ratio,
            "制动响应时间": self.response_time_s,
            "线路坡度": self.grade_permille,
            "基本运行阻力系数": self.basic_resistance_coefficient,
            "旋转质量换算系数": self.rotating_mass_factor,
            "轮轨黏着系数": self.adhesion_coefficient,
        }
        for label, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{label}必须是有限数值。")

        if self.mass_kg <= 0:
            raise ValueError("车辆质量必须大于 0 kg。")
        if self.initial_speed_kmh < 0:
            raise ValueError("初速度不能小于 0 km/h。")
        if not 0 < self.braking_ratio <= 1:
            raise ValueError("制动率必须大于 0 且不超过 1。")
        if self.response_time_s < 0:
            raise ValueError("制动响应时间不能小于 0 s。")
        if self.basic_resistance_coefficient < 0:
            raise ValueError("基本运行阻力系数不能小于 0。")
        if self.rotating_mass_factor < 0:
            raise ValueError("旋转质量换算系数不能小于 0。")
        if not 0 < self.adhesion_coefficient <= 1:
            raise ValueError("轮轨黏着系数必须大于 0 且不超过 1。")


@dataclass(frozen=True)
class CurvePoint:
    time_s: float
    distance_m: float
    speed_m_s: float


@dataclass(frozen=True)
class BrakingResult:
    case: BrakingCase
    deceleration_m_s2: float
    braking_force_n: float
    adhesion_limit_force_n: float
    response_distance_m: float
    braking_distance_m: float
    stopping_distance_m: float
    braking_time_s: float
    total_time_s: float
    curve_points: tuple[CurvePoint, ...]
