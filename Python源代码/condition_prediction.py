"""UI for the synthetic-only condition limit demonstration."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

from condition_cnn import (
    CLASS_NAMES,
    SYNTHETIC_TRAINING_NOTICE,
    SyntheticConditionCNN,
    TrainingSummary,
    train_synthetic_model,
)
from models import BrakingCase


class ConditionPredictionWindow:
    def __init__(self, parent: tk.Misc, case: BrakingCase) -> None:
        self.case = case
        self.window = tk.Toplevel(parent)
        self.window.title("合成数据 CNN 车辆耐受等级演示")
        self.window.geometry("760x540")
        self.window.minsize(620, 440)
        self.window.transient(parent)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)

        ttk.Label(
            self.window,
            text="车辆状况与耐受等级预测（合成数据教学演示）",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        notice = (
            "重要：模型将现场生成虚构时间序列并训练 1D CNN。"
            "标签阈值不来自材料、部件试验或有限元；输出不能预测真实损伤、"
            "耐受极限或车辆散架。"
        )
        ttk.Label(
            self.window,
            text=notice,
            wraplength=700,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=5)

        self.report = tk.Text(
            self.window,
            height=16,
            wrap="word",
            state="disabled",
            font=("Microsoft YaHei UI", 9),
        )
        self.report.grid(row=2, column=0, sticky="nsew", padx=14, pady=6)

        self.status = tk.StringVar(value="正在生成合成数据并训练卷积网络……")
        ttk.Label(
            self.window, textvariable=self.status, anchor="w"
        ).grid(row=3, column=0, sticky="ew", padx=14, pady=4)
        self.close_button = ttk.Button(
            self.window, text="关闭", command=self.window.destroy, state="disabled"
        )
        self.close_button.grid(row=4, column=0, sticky="e", padx=14, pady=(4, 12))

        self._results: queue.Queue[
            tuple[SyntheticConditionCNN, TrainingSummary] | Exception
        ] = queue.Queue()
        self._show_report("生成 2,400 条合成时序数据，并训练 1D 卷积分类器。")
        self._worker = threading.Thread(
            target=self._train, name="synthetic-cnn-training", daemon=True
        )
        self._worker.start()
        self.window.after(100, self._poll_training)

    def _train(self) -> None:
        try:
            self._results.put(train_synthetic_model(sample_count=2400, seed=7))
        except (FloatingPointError, OverflowError, ValueError) as error:
            self._results.put(error)

    def _poll_training(self) -> None:
        if self.window.winfo_exists() is False:
            return
        try:
            outcome = self._results.get_nowait()
        except queue.Empty:
            self.window.after(100, self._poll_training)
            return

        self.close_button.configure(state="normal")
        if isinstance(outcome, Exception):
            self.status.set("训练或预测失败。")
            self._show_report(f"未生成预测结果：{outcome}")
            return

        model, summary = outcome
        try:
            prediction = model.predict(self.case)
        except (OverflowError, ValueError) as error:
            self.status.set("当前参数无法转换为有限的模型输入。")
            self._show_report(f"未生成预测结果：{error}")
            return

        self.status.set("合成数据训练与演示预测完成。")
        self._show_report(
            "\n".join(
                [
                    f"预测类别：{prediction.class_name}",
                    f"演示类别置信度（非真实概率）：{prediction.confidence:.1%}",
                    f"演示合成负荷指数：{prediction.synthetic_load_index:.3f}",
                    f"合成留出集准确率：{summary.validation_accuracy:.1%} "
                    f"（{summary.sample_count} 条合成样本，"
                    f"{summary.epochs} 轮训练）",
                    "网络：可训练 1D 卷积层（12 个卷积核、核宽 5）+ "
                    "全局平均池化 + softmax 分类。",
                    "合成等级阈值：负荷指数 <0.75、0.75–1.35、"
                    "1.35–2.50、≥2.50；这些阈值仅用于生成教学标签。",
                    "演示负荷指数公式：0.16×质量/40000 + "
                    "0.28×(速度/80)^2 + 0.18×制动率/0.12 + "
                    "0.12×|坡度|/30 + 0.26×制动率/黏着系数。",
                    "",
                    "类别概率（仅表示模型对虚构标签的输出）：",
                    *[
                        f"- {name}：{probability:.1%}"
                        for name, probability in zip(
                            CLASS_NAMES, prediction.probabilities
                        )
                    ],
                    "",
                    "当前输入参数：",
                    f"- 车辆质量：{self.case.mass_kg:.4g} kg",
                    f"- 初速度：{self.case.initial_speed_kmh:.4g} km/h",
                    f"- 制动率：{self.case.braking_ratio:.4g}",
                    f"- 线路坡度：{self.case.grade_permille:.4g}‰",
                    f"- 轮轨黏着系数：{self.case.adhesion_coefficient:.4g}",
                    "",
                    "解释：分类阈值由合成负荷指数生成，不表示任何具体零件、"
                    "材料或整车的实测耐受极限。极端类别仅是软件生成的教学标签，"
                    "不表示车辆将会散架。",
                    "",
                    SYNTHETIC_TRAINING_NOTICE,
                ]
            )
        )

    def _show_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")
