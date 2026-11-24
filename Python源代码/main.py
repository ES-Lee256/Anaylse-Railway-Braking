"""Railway vehicle braking performance analysis application."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bearing_cnn import BEARING_CLASS_NAMES, MODEL_PATH, predict_bearing_file
from calculations import calculate_braking
from charts import draw_braking_chart
from condition_prediction import ConditionPredictionWindow
from condition_cnn import Trainable1DCNN
from data_io import export_results, load_cases
from models import BrakingCase, BrakingResult
from simulation import SimulationWindow

DEFAULT_VALUES = {
    "name": "示例车辆",
    "mass_kg": "40000",
    "initial_speed_kmh": "80",
    "braking_ratio": "0.12",
    "response_time_s": "1.0",
    "grade_permille": "0",
    "basic_resistance_coefficient": "0.002",
    "rotating_mass_factor": "0.06",
    "adhesion_coefficient": "0.25",
}

INPUTS = (
    ("name", "工况名称", None),
    ("mass_kg", "车辆质量 (kg)", "40000"),
    ("initial_speed_kmh", "初速度 (km/h)", "80"),
    ("braking_ratio", "制动率 β", "0.12"),
    ("response_time_s", "制动响应时间 (s)", "1.0"),
    ("grade_permille", "线路坡度 (‰，上坡为正)", "0"),
    ("basic_resistance_coefficient", "基本运行阻力系数", "0.002"),
    ("rotating_mass_factor", "旋转质量换算系数", "0.06"),
    ("adhesion_coefficient", "轮轨黏着系数 μ", "0.25"),
)

RESULT_COLUMNS = (
    ("name", "工况", 110),
    ("deceleration", "减速度 (m/s²)", 110),
    ("response_distance", "响应距离 (m)", 105),
    ("braking_distance", "制动距离 (m)", 105),
    ("stopping_distance", "总停车距离 (m)", 115),
    ("braking_time", "制动时间 (s)", 100),
    ("total_time", "总时间 (s)", 95),
)


class BrakingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("铁路车辆制动性能计算与分析 V2")
        self.root.minsize(1000, 700)
        self.entries: dict[str, ttk.Entry] = {}
        self.results: list[BrakingResult] = []
        self.chart_results: list[BrakingResult] = []
        self._build_interface()
        self._draw_empty_chart()

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        input_frame = ttk.LabelFrame(self.root, text="车辆与工况参数")
        input_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        for column in range(6):
            input_frame.columnconfigure(column, weight=1)

        for index, (key, label, default) in enumerate(INPUTS):
            row, column = divmod(index, 3)
            cell = ttk.Frame(input_frame)
            cell.grid(row=row, column=column, sticky="ew", padx=8, pady=5)
            ttk.Label(cell, text=label).pack(anchor="w")
            entry = ttk.Entry(cell)
            entry.pack(fill="x", pady=(2, 0))
            value = DEFAULT_VALUES[key] if default is None else default
            entry.insert(0, value)
            self.entries[key] = entry

        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        ttk.Button(button_frame, text="计算当前参数", command=self.calculate_current).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(
            button_frame,
            text="演示当前参数",
            command=self.demonstrate_current,
        ).pack(side="left", padx=6)
        ttk.Button(
            button_frame,
            text="预测车辆状况（演示）",
            command=self.predict_condition,
        ).pack(side="left", padx=6)
        ttk.Button(
            button_frame,
            text="识别轴承状态（实测数据模型）",
            command=self.predict_bearing_data,
        ).pack(side="left", padx=6)
        ttk.Button(button_frame, text="导入 CSV 工况", command=self.import_cases).pack(
            side="left", padx=6
        )
        ttk.Button(button_frame, text="导出计算结果", command=self.export_current).pack(
            side="left", padx=6
        )
        ttk.Button(button_frame, text="恢复默认值", command=self.reset).pack(
            side="left", padx=6
        )

        content = ttk.Frame(self.root)
        content.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        table_frame = ttk.LabelFrame(content, text="计算结果")
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_frame,
            columns=tuple(column[0] for column in RESULT_COLUMNS),
            show="headings",
            height=12,
        )
        for key, title, width in RESULT_COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=75, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        chart_frame = ttk.LabelFrame(content, text="曲线分析")
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(1, weight=1)
        chart_controls = ttk.Frame(chart_frame)
        chart_controls.grid(row=0, column=0, sticky="ew", padx=6, pady=5)
        ttk.Label(chart_controls, text="工况").pack(side="left")
        self.case_selector = ttk.Combobox(
            chart_controls, state="readonly", width=24
        )
        self.case_selector.pack(side="left", padx=(5, 12))
        self.case_selector.bind("<<ComboboxSelected>>", self._refresh_chart)
        ttk.Label(chart_controls, text="曲线").pack(side="left")
        self.chart_selector = ttk.Combobox(
            chart_controls,
            state="readonly",
            values=("速度—时间", "速度—距离"),
            width=14,
        )
        self.chart_selector.current(0)
        self.chart_selector.pack(side="left", padx=5)
        self.chart_selector.bind("<<ComboboxSelected>>", self._refresh_chart)

        self.figure = Figure(figsize=(5.4, 3.8), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        self.status = tk.StringVar(
            value="教学用简化模型：上坡坡度为正；结果不用于铁路运营或安全决策。"
        )
        ttk.Label(
            self.root,
            textvariable=self.status,
            wraplength=1100,
            justify="left",
        ).grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 12))

    def _case_from_entries(self) -> BrakingCase:
        try:
            return BrakingCase(
                name=self.entries["name"].get().strip(),
                mass_kg=float(self.entries["mass_kg"].get()),
                initial_speed_kmh=float(self.entries["initial_speed_kmh"].get()),
                braking_ratio=float(self.entries["braking_ratio"].get()),
                response_time_s=float(self.entries["response_time_s"].get()),
                grade_permille=float(self.entries["grade_permille"].get()),
                basic_resistance_coefficient=float(
                    self.entries["basic_resistance_coefficient"].get()
                ),
                rotating_mass_factor=float(
                    self.entries["rotating_mass_factor"].get()
                ),
                adhesion_coefficient=float(
                    self.entries["adhesion_coefficient"].get()
                ),
            )
        except ValueError as error:
            raise ValueError(f"请检查参数输入：{error}") from error

    def calculate_current(self) -> None:
        try:
            case = self._case_from_entries()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error), parent=self.root)
            return
        self._calculate_cases([case], source="当前参数")

    def demonstrate_current(self) -> None:
        try:
            case = self._case_from_entries()
            SimulationWindow(self.root, case)
        except ValueError as error:
            messagebox.showerror("无法开始演示", str(error), parent=self.root)

    def predict_condition(self) -> None:
        try:
            case = self._case_from_entries()
            ConditionPredictionWindow(self.root, case)
        except ValueError as error:
            messagebox.showerror("无法开始预测", str(error), parent=self.root)

    def predict_bearing_data(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择轴承振动 MATLAB 数据",
            filetypes=(("MATLAB 数据文件", "*.mat"), ("所有文件", "*.*")),
            initialdir=str(Path(__file__).resolve().parent / "data"),
        )
        if not path:
            return

        application_root = Path(
            getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
        )
        model_path = application_root / MODEL_PATH.relative_to(
            Path(__file__).resolve().parent
        )
        if not model_path.is_file():
            messagebox.showerror(
                "模型文件不存在",
                "未找到实测数据 CNN 模型。请先运行 train_bearing_cnn.py 完成训练。",
                parent=self.root,
            )
            return

        try:
            model = Trainable1DCNN.load(model_path)
            class_index, probabilities = predict_bearing_file(model, Path(path))
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("轴承数据识别失败", str(error), parent=self.root)
            return

        scores = "\n".join(
            f"{name}：{score:.1%}"
            for name, score in zip(BEARING_CLASS_NAMES, probabilities)
        )
        messagebox.showinfo(
            "轴承状态分类结果",
            f"预测类别：{BEARING_CLASS_NAMES[class_index]}\n"
            f"各类别平均 softmax 分数：\n{scores}\n\n"
            "模型使用 Mendeley Data 实验台轴承振动记录训练。"
            "模型仅覆盖 200 kHz、10 s 的转速递增记录；数据并非铁路车辆轴箱/轮对数据。"
            "分数未校准，不代表真实故障概率，"
            "不可用于车辆安全或维修决策。",
            parent=self.root,
        )

    def import_cases(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择工况 CSV 文件",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
            initialdir=str(Path.cwd()),
        )
        if not path:
            return
        try:
            cases = load_cases(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("导入失败", str(error), parent=self.root)
            return
        self._calculate_cases(cases, source=f"导入 {len(cases)} 组工况")

    def _calculate_cases(self, cases: list[BrakingCase], source: str) -> None:
        results: list[BrakingResult] = []
        errors: list[str] = []
        for case in cases:
            try:
                results.append(calculate_braking(case))
            except ValueError as error:
                errors.append(str(error))

        self.results = results
        self.chart_results = results
        self._populate_results()
        self._populate_chart_selector()
        if results:
            self.status.set(
                f"{source}：成功计算 {len(results)} 组；"
                f"简化模型按恒定减速度计算。"
            )
        else:
            self.status.set(f"{source}：没有工况通过计算，请查看错误提示。")

        if errors:
            details = "\n".join(f"• {error}" for error in errors[:8])
            if len(errors) > 8:
                details += f"\n另有 {len(errors) - 8} 项错误未显示。"
            if results:
                messagebox.showwarning(
                    "部分工况未计算",
                    f"成功 {len(results)} 组，失败 {len(errors)} 组：\n{details}",
                    parent=self.root,
                )
            else:
                messagebox.showerror(
                    "计算失败", details, parent=self.root
                )

    def _populate_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for result in self.results:
            self.tree.insert(
                "",
                "end",
                values=(
                    result.case.name,
                    f"{result.deceleration_m_s2:.3f}",
                    f"{result.response_distance_m:.2f}",
                    f"{result.braking_distance_m:.2f}",
                    f"{result.stopping_distance_m:.2f}",
                    f"{result.braking_time_s:.2f}",
                    f"{result.total_time_s:.2f}",
                ),
            )

    def _populate_chart_selector(self) -> None:
        labels = [
            f"{index + 1}. {result.case.name}"
            for index, result in enumerate(self.chart_results)
        ]
        self.case_selector.configure(values=labels)
        if labels:
            self.case_selector.current(0)
            self._refresh_chart()
        else:
            self.case_selector.set("")
            self._draw_empty_chart()

    def _refresh_chart(self, _event: object = None) -> None:
        selected = self.case_selector.current()
        if not 0 <= selected < len(self.chart_results):
            return
        chart_type = self.chart_selector.get()
        draw_braking_chart(
            self.axes, self.chart_results[selected], chart_type
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_empty_chart(self) -> None:
        self.axes.clear()
        self.axes.set_title("计算后显示制动速度曲线")
        self.axes.set_xlabel("时间 (s)")
        self.axes.set_ylabel("速度 (km/h)")
        self.axes.grid(True, linestyle="--", alpha=0.4)
        self.canvas.draw_idle()

    def export_current(self) -> None:
        if not self.results:
            messagebox.showerror("无法导出", "当前没有可导出的计算结果。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出计算结果",
            defaultextension=".csv",
            initialfile="制动性能计算结果.csv",
            filetypes=(("CSV 文件", "*.csv"),),
        )
        if not path:
            return
        try:
            export_results(path, self.results)
        except OSError as error:
            messagebox.showerror("导出失败", str(error), parent=self.root)
            return
        self.status.set(f"计算结果已导出：{path}")
        messagebox.showinfo("导出完成", "计算结果已成功导出。", parent=self.root)

    def reset(self) -> None:
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, DEFAULT_VALUES[key])
        self.results = []
        self.chart_results = []
        self._populate_results()
        self._populate_chart_selector()
        self.status.set(
            "已恢复默认参数。教学用简化模型：上坡坡度为正；结果不用于铁路运营或安全决策。"
        )


def main() -> None:
    root = tk.Tk()
    BrakingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
