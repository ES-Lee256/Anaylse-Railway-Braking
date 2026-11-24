"""Simplified runaway-risk assessment for the railway vehicle demo."""

from dataclasses import dataclass
import math

from calculations import GRAVITY_M_S2
from models import BrakingCase

MAX_DEMO_TIME_S = 120.0
RUNAWAY_DEMO_TAIL_S = 8.0


@dataclass(frozen=True)
class RunawayAssessment:
    case: BrakingCase
    initial_speed_m_s: float
    effective_braking_ratio: float
    net_deceleration_m_s2: float
    adhesion_exceeded: bool
    classification: str
    findings: tuple[str, ...]
    stopping_time_s: float | None
    stopping_distance_m: float | None
    simulation_duration_s: float
    simulation_time_limited: bool

    def state_at(self, time_s: float) -> tuple[float, float]:
        """Return simulated distance and speed at elapsed time."""
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("仿真时间必须是有限的非负数。")

        speed = self.initial_speed_m_s
        if speed == 0:
            return 0.0, 0.0

        response_time = min(time_s, self.case.response_time_s)
        distance = speed * response_time
        braking_elapsed = max(0.0, time_s - self.case.response_time_s)
        deceleration = self.net_deceleration_m_s2

        if deceleration > 0:
            braking_elapsed = min(
                braking_elapsed, speed / deceleration
            )
            distance += (
                speed * braking_elapsed
                - 0.5 * deceleration * braking_elapsed**2
            )
            speed = max(0.0, speed - deceleration * braking_elapsed)
        else:
            distance += (
                speed * braking_elapsed
                - 0.5 * deceleration * braking_elapsed**2
            )
            speed -= deceleration * braking_elapsed

        if not math.isfinite(distance) or not math.isfinite(speed):
            raise ValueError("当前参数导致仿真超出数值范围。")
        return distance, speed


def assess_runaway(case: BrakingCase) -> RunawayAssessment:
    """Assess adhesion exceedance and whether the simplified model can stop."""
    initial_speed_m_s = case.initial_speed_kmh / 3.6
    adhesion_exceeded = case.braking_ratio > case.adhesion_coefficient
    effective_braking_ratio = min(
        case.braking_ratio, case.adhesion_coefficient
    )
    net_deceleration = (
        GRAVITY_M_S2
        * (
            effective_braking_ratio
            + case.basic_resistance_coefficient
            + case.grade_permille / 1000
        )
        / (1 + case.rotating_mass_factor)
    )
    response_distance = initial_speed_m_s * case.response_time_s
    if not all(
        math.isfinite(value)
        for value in (initial_speed_m_s, net_deceleration, response_distance)
    ):
        raise ValueError("当前参数导致仿真超出数值范围。")

    findings: list[str] = []
    stopping_time: float | None = None
    stopping_distance: float | None = None

    if adhesion_exceeded:
        findings.append(
            "设定制动率超过轮轨黏着系数；仿真按黏着上限降低有效制动率，"
            "存在制动能力受限和滑移风险。"
        )

    if initial_speed_m_s == 0:
        classification = (
            "静止状态（参数存在黏着超限）"
            if adhesion_exceeded
            else "静止状态"
        )
        requested_duration = max(case.response_time_s, 1.0)
    elif net_deceleration <= 0:
        findings.append(
            "按黏着限值修正后，当前坡度和制动参数不能形成正的制动减速度；"
            "简化模型判定车辆不能停车，演示中将持续运行。"
        )
        classification = (
            "失控风险（黏着受限且制动能力不足）"
            if adhesion_exceeded
            else "失控风险（制动能力不足）"
        )
        requested_duration = case.response_time_s + RUNAWAY_DEMO_TAIL_S
    else:
        try:
            braking_time = initial_speed_m_s / net_deceleration
            braking_distance = initial_speed_m_s**2 / (2 * net_deceleration)
        except OverflowError as error:
            raise ValueError("当前参数导致仿真超出数值范围。") from error
        stopping_time = case.response_time_s + braking_time
        stopping_distance = response_distance + braking_distance
        if not all(
            math.isfinite(value)
            for value in (braking_time, braking_distance, stopping_time, stopping_distance)
        ):
            raise ValueError("当前参数导致仿真超出数值范围。")
        classification = (
            "黏着受限（可停车但存在制动风险）"
            if adhesion_exceeded
            else "模型内可控停车"
        )
        requested_duration = stopping_time

    if not findings:
        findings.append(
            "输入参数满足本演示采用的简化黏着条件，且模型可形成正的制动减速度。"
        )

    return RunawayAssessment(
        case=case,
        initial_speed_m_s=initial_speed_m_s,
        effective_braking_ratio=effective_braking_ratio,
        net_deceleration_m_s2=net_deceleration,
        adhesion_exceeded=adhesion_exceeded,
        classification=classification,
        findings=tuple(findings),
        stopping_time_s=stopping_time,
        stopping_distance_m=stopping_distance,
        simulation_duration_s=min(requested_duration, MAX_DEMO_TIME_S),
        simulation_time_limited=requested_duration > MAX_DEMO_TIME_S,
    )
