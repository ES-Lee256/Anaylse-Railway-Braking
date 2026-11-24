"""Tkinter animation and final runaway-risk report."""

import time
import tkinter as tk
from tkinter import ttk

from models import BrakingCase
from runaway import RunawayAssessment, assess_runaway

ANIMATION_DURATION_MS = 7000
FRAME_INTERVAL_MS = 40


class SimulationWindow:
    def __init__(self, parent: tk.Misc, case: BrakingCase) -> None:
        self.assessment = assess_runaway(case)
        self.window = tk.Toplevel(parent)
        self.window.title(f"车辆制动演示与失控检测报告 - {case.name}")
        self.window.geometry("920x600")
        self.window.minsize(760, 520)
        self.window.transient(parent)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        ttk.Label(
            self.window,
            text=f"当前参数模拟演示：{case.name}",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self.canvas = tk.Canvas(
            self.window,
            height=300,
            background="#e8f4fc",
            highlightthickness=1,
            highlightbackground="#aab7c4",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=6)
        self._draw_scene()

        self.status = tk.StringVar(value="正在播放加速的车辆运动模拟……")
        ttk.Label(
            self.window, textvariable=self.status, anchor="w"
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=4)

        report_frame = ttk.LabelFrame(self.window, text="失控检测报告")
        report_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 12))
        report_frame.columnconfigure(0, weight=1)
        self.report = tk.Text(
            report_frame,
            height=8,
            wrap="word",
            state="disabled",
            font=("Microsoft YaHei UI", 9),
        )
        self.report.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self._set_report("动画结束后将在此显示检测依据与结果。")

        self._started_at = time.perf_counter()
        self._wall_duration_s = (
            ANIMATION_DURATION_MS / 1000
            if self.assessment.simulation_duration_s > 0
            else 1.0
        )
        self._end_distance_m, _ = self.assessment.state_at(
            self.assessment.simulation_duration_s
        )
        self._animate()

    def _draw_scene(self) -> None:
        self.canvas.create_oval(
            730, 30, 790, 90, fill="#ffd166", outline=""
        )
        self.canvas.create_rectangle(
            0, 205, 920, 300, fill="#cfe6c8", outline=""
        )
        self.canvas.create_line(
            0, 235, 920, 235, fill="#737b83", width=5
        )
        self.canvas.create_line(
            0, 250, 920, 250, fill="#737b83", width=5
        )
        for x in range(0, 920, 38):
            self.canvas.create_line(
                x, 226, x + 12, 259, fill="#795548", width=5
            )

        x, y = 55, 142
        self.canvas.create_rectangle(
            x, y + 24, x + 168, y + 82,
            fill="#1565c0", outline="#0d3d74", width=2, tags="vehicle"
        )
        self.canvas.create_polygon(
            x + 168, y + 24, x + 194, y + 42, x + 194, y + 82,
            x + 168, y + 82,
            fill="#1976d2", outline="#0d3d74", width=2, tags="vehicle"
        )
        self.canvas.create_rectangle(
            x + 13, y, x + 155, y + 28,
            fill="#e53935", outline="#9e2020", width=2, tags="vehicle"
        )
        self.canvas.create_rectangle(
            x + 28, y + 35, x + 54, y + 55,
            fill="#b3e5fc", outline="#0d3d74", tags="vehicle"
        )
        self.canvas.create_rectangle(
            x + 68, y + 35, x + 94, y + 55,
            fill="#b3e5fc", outline="#0d3d74", tags="vehicle"
        )
        self.canvas.create_rectangle(
            x + 108, y + 35, x + 134, y + 55,
            fill="#b3e5fc", outline="#0d3d74", tags="vehicle"
        )
        self.canvas.create_oval(
            x + 28, y + 70, x + 62, y + 104,
            fill="#263238", outline="#111", tags="vehicle"
        )
        self.canvas.create_oval(
            x + 135, y + 70, x + 169, y + 104,
            fill="#263238", outline="#111", tags="vehicle"
        )
        self.distance_label = self.canvas.create_text(
            18, 18, anchor="nw", font=("Microsoft YaHei UI", 10)
        )
        self.speed_label = self.canvas.create_text(
            18, 42, anchor="nw", font=("Microsoft YaHei UI", 10)
        )

    def _animate(self) -> None:
        elapsed_wall = time.perf_counter() - self._started_at
        progress = min(1.0, elapsed_wall / self._wall_duration_s)
        simulated_time = (
            self.assessment.simulation_duration_s * progress
        )
        distance_m, speed_m_s = self.assessment.state_at(simulated_time)
        self._place_vehicle(distance_m)
        self.canvas.itemconfigure(
            self.distance_label, text=f"模拟距离：{distance_m:.1f} m"
        )
        self.canvas.itemconfigure(
            self.speed_label, text=f"当前速度：{speed_m_s * 3.6:.1f} km/h"
        )
        self.status.set(
            f"模拟时间 {simulated_time:.1f} s / "
            f"{self.assessment.simulation_duration_s:.1f} s"
        )

        if progress >= 1.0:
            self._finish(distance_m, speed_m_s)
        else:
            self.window.after(FRAME_INTERVAL_MS, self._animate)

    def _place_vehicle(self, distance_m: float) -> None:
        start_x = 55.0
        travel_width = 610.0
        if self._end_distance_m > 0:
            fraction = min(1.0, distance_m / self._end_distance_m)
        else:
            fraction = 0.0
        target_x = start_x + travel_width * fraction
        current = self.canvas.bbox("vehicle")
        if current is None:
            return
        self.canvas.move("vehicle", target_x - current[0], 0)

    def _finish(self, distance_m: float, speed_m_s: float) -> None:
        if speed_m_s <= 1e-6 and self.assessment.stopping_time_s is not None:
            outcome = "演示车辆在当前简化模型中已停车。"
        elif self.assessment.case.initial_speed_kmh == 0:
            outcome = "初速度为零，车辆保持静止。"
        else:
            outcome = "演示时间结束时车辆仍在运动，未观察到停车。"
        self.status.set(
            f"演示结束：模拟距离 {distance_m:.1f} m，"
            f"当前速度 {speed_m_s * 3.6:.1f} km/h。"
        )
        self._set_report(self._build_report(distance_m, speed_m_s, outcome))

    def _build_report(
        self, distance_m: float, speed_m_s: float, outcome: str
    ) -> str:
        assessment = self.assessment
        case = assessment.case
        lines = [
            f"检测结论：{assessment.classification}",
            f"工况：{case.name}；初速度：{case.initial_speed_kmh:.3g} km/h；"
            f"坡度：{case.grade_permille:.3g}‰（上坡为正）",
            f"设定制动率 β={case.braking_ratio:.4g}，"
            f"黏着系数 μ={case.adhesion_coefficient:.4g}，"
            f"仿真有效制动率={assessment.effective_braking_ratio:.4g}",
            f"模型等效减速度：{assessment.net_deceleration_m_s2:.4g} m/s²",
        ]
        if assessment.stopping_time_s is not None:
            lines.append(
                f"模型预计总时间：{assessment.stopping_time_s:.2f} s；"
                f"预计总停车距离：{assessment.stopping_distance_m:.2f} m"
            )
        for finding in assessment.findings:
            lines.append(f"检测依据：{finding}")
        lines.extend(
            [
                f"动画终态：{outcome} 模拟距离 {distance_m:.2f} m，"
                f"速度 {speed_m_s * 3.6:.2f} km/h。",
                "说明：动画为压缩时长的教学示意，检测仅依据简化模型，"
                "不代表实际车辆安全评估或运营决策。",
            ]
        )
        if assessment.simulation_time_limited:
            lines.append("提示：模拟时间达到演示上限，未完整播放至模型停车时刻。")
        return "\n".join(lines)

    def _set_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")
