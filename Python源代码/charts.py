"""Plotting helpers for braking analysis results."""

from matplotlib import rcParams
from matplotlib.axes import Axes

from models import BrakingResult

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def draw_braking_chart(
    axes: Axes, result: BrakingResult, chart_type: str
) -> None:
    axes.clear()
    if chart_type == "速度—时间":
        x_values = [point.time_s for point in result.curve_points]
        y_values = [point.speed_m_s * 3.6 for point in result.curve_points]
        axes.plot(x_values, y_values, color="#1565c0", linewidth=2)
        axes.set_xlabel("时间 (s)")
        axes.set_ylabel("速度 (km/h)")
        axes.set_title(f"{result.case.name}：速度—时间曲线")
    elif chart_type == "速度—距离":
        x_values = [point.distance_m for point in result.curve_points]
        y_values = [point.speed_m_s * 3.6 for point in result.curve_points]
        axes.plot(x_values, y_values, color="#2e7d32", linewidth=2)
        axes.set_xlabel("距离 (m)")
        axes.set_ylabel("速度 (km/h)")
        axes.set_title(f"{result.case.name}：速度—距离曲线")
    else:
        raise ValueError(f"不支持的曲线类型：{chart_type}")

    axes.grid(True, linestyle="--", alpha=0.4)
    axes.margins(x=0.03, y=0.08)
