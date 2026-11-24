"""Download the public bearing dataset, train a grouped-split CNN, and report metrics."""

from __future__ import annotations

from pathlib import Path

from bearing_cnn import (
    BEARING_CLASS_NAMES,
    MODEL_PATH,
    train_bearing_model,
)
from download_bearing_dataset import DATA_DIRECTORY, download_selected_files

REPORT_PATH = Path(__file__).resolve().parent / "轴承CNN训练报告.txt"


def main() -> None:
    download_selected_files(DATA_DIRECTORY)
    result = train_bearing_model(DATA_DIRECTORY, MODEL_PATH)
    lines = [
        "Mendeley Data 轴承振动数据集 CNN 训练报告",
        "数据集：Bearing Vibration Data under Time-varying Rotational Speed Conditions",
        "DOI：10.17632/v43hmbwxpm.2",
        "许可：CC BY 4.0；作者 Huan Huang、Natalie Baddour；机构 University of Ottawa。",
        "使用范围：仅选取 A（转速递增）工况下的 15 条原始试验记录。",
        "训练/验证划分：每类试验 01、02 用于训练；完整试验 03 留作验证，按试验记录分组，未混切同一试验。",
        "预处理：Channel_1 加速度信号经抗混叠多相滤波降采样 10 倍；512 点窗长、2048 点步长；每窗包含标准化波形及 log10(RMS) 通道，通道归一化参数只由训练集估计。",
        "网络：NumPy 实现的 1D 卷积（12 个卷积核、核宽 5）+ ReLU + 全局均值池化 + softmax；80 轮训练。",
        f"训练窗口：{result.training_sample_count}",
        f"独立验证窗口：{result.validation_sample_count}",
        f"验证准确率：{result.summary.validation_accuracy:.2%}",
        "",
        "类别及验证指标（precision / recall）：",
    ]
    for index, name in enumerate(BEARING_CLASS_NAMES):
        lines.append(
            f"- {name}：{result.precision_by_class[index]:.2%} / "
            f"{result.recall_by_class[index]:.2%}"
        )
    lines.extend(["", "混淆矩阵（行=真实类别，列=预测类别）："])
    lines.extend(
        "  " + " ".join(str(value) for value in row)
        for row in result.confusion_matrix
    )
    lines.extend(
        [
            "",
            f"模型文件：{MODEL_PATH.relative_to(Path(__file__).resolve().parent)}",
            "",
            "限制：该数据是实验台轴承振动数据，并非铁路车辆轴箱/轮对数据；只覆盖上述数据与工况。"
            "验证结果不代表铁路车辆实际诊断精度，不能推断整车损坏、材料耐受极限或散架风险。",
            "软最大值仅为模型分数，不是经过校准的真实故障概率。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
