"""Calculation functions for the simplified railway vehicle braking model."""

import math

from models import BrakingCase, BrakingResult, CurvePoint

GRAVITY_M_S2 = 9.80665
DEFAULT_CURVE_SAMPLES = 100


def calculate_braking(
    case: BrakingCase, samples: int = DEFAULT_CURVE_SAMPLES
) -> BrakingResult:
    """Calculate stopping performance and curve points for one operating case."""
    if samples < 2:
        raise ValueError("曲线采样点数必须不少于 2。")
    if case.braking_ratio > case.adhesion_coefficient:
        raise ValueError(
            f"工况“{case.name}”的制动率高于输入的轮轨黏着系数，"
            "简化模型不计算该工况。"
        )

    initial_speed_m_s = case.initial_speed_kmh / 3.6
    deceleration_m_s2 = (
        GRAVITY_M_S2
        * (
            case.braking_ratio
            + case.basic_resistance_coefficient
            + case.grade_permille / 1000
        )
        / (1 + case.rotating_mass_factor)
    )
    if initial_speed_m_s > 0 and deceleration_m_s2 <= 0:
        raise ValueError(
            f"工况“{case.name}”的计算减速度不大于 0，"
            "该工况无法在当前模型下停车。"
        )

    response_time_s = case.response_time_s if initial_speed_m_s > 0 else 0.0
    response_distance_m = initial_speed_m_s * response_time_s
    braking_time_s = (
        initial_speed_m_s / deceleration_m_s2 if initial_speed_m_s > 0 else 0.0
    )
    braking_distance_m = (
        initial_speed_m_s**2 / (2 * deceleration_m_s2)
        if initial_speed_m_s > 0
        else 0.0
    )
    total_time_s = response_time_s + braking_time_s
    stopping_distance_m = response_distance_m + braking_distance_m
    braking_force_n = case.braking_ratio * case.mass_kg * GRAVITY_M_S2
    adhesion_limit_force_n = (
        case.adhesion_coefficient * case.mass_kg * GRAVITY_M_S2
    )
    if not all(
        math.isfinite(value)
        for value in (
            deceleration_m_s2,
            response_distance_m,
            braking_distance_m,
            stopping_distance_m,
            braking_time_s,
            total_time_s,
            braking_force_n,
            adhesion_limit_force_n,
        )
    ):
        raise ValueError(f"工况“{case.name}”的计算结果超出数值范围。")

    curve_points = _build_curve_points(
        initial_speed_m_s,
        response_time_s,
        deceleration_m_s2,
        braking_time_s,
        response_distance_m,
        samples,
    )
    return BrakingResult(
        case=case,
        deceleration_m_s2=deceleration_m_s2,
        braking_force_n=braking_force_n,
        adhesion_limit_force_n=adhesion_limit_force_n,
        response_distance_m=response_distance_m,
        braking_distance_m=braking_distance_m,
        stopping_distance_m=stopping_distance_m,
        braking_time_s=braking_time_s,
        total_time_s=total_time_s,
        curve_points=curve_points,
    )


def _build_curve_points(
    initial_speed_m_s: float,
    response_time_s: float,
    deceleration_m_s2: float,
    braking_time_s: float,
    response_distance_m: float,
    samples: int,
) -> tuple[CurvePoint, ...]:
    points = [CurvePoint(0.0, 0.0, initial_speed_m_s)]

    if response_time_s > 0:
        for index in range(1, samples + 1):
            elapsed = response_time_s * index / samples
            points.append(
                CurvePoint(
                    elapsed,
                    initial_speed_m_s * elapsed,
                    initial_speed_m_s,
                )
            )

    if braking_time_s > 0:
        for index in range(1, samples + 1):
            elapsed = braking_time_s * index / samples
            speed = max(0.0, initial_speed_m_s - deceleration_m_s2 * elapsed)
            distance = (
                response_distance_m
                + initial_speed_m_s * elapsed
                - 0.5 * deceleration_m_s2 * elapsed**2
            )
            points.append(
                CurvePoint(
                    response_time_s + elapsed,
                    distance,
                    speed,
                )
            )
    return tuple(points)
