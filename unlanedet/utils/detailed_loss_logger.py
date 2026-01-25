#!/usr/bin/env python3
"""
分项损失详细记录器

专门记录 XYTL 回归损失的各个分项，用于监控和可视化
不影响训练流程，仅记录额外的监控数据

使用方法：
    from unlanedet.utils.detailed_loss_logger import DetailedLossLogger

    logger = DetailedLossLogger(output_dir)
    logger.log(iteration, loss_dict)
"""

import json
import os
from datetime import datetime
import logging


class DetailedLossLogger:
    """分项损失详细记录器"""

    def __init__(self, output_dir, filename="detailed_metrics.json"):
        """
        初始化记录器

        Args:
            output_dir: 输出目录路径
            filename: 日志文件名（默认 detailed_metrics.json）
        """
        self.output_dir = output_dir
        self.filename = filename
        self.file_path = os.path.join(output_dir, filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 如果文件不存在，创建空文件
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                pass  # 创建空文件

        # 内部计数器，用于自动生成迭代序号
        self._counter = 0

    def log(self, iteration=None, losses_dict=None):
        """
        记录分项损失到 JSON Lines 文件

        Args:
            iteration: 当前迭代次数（可选，如果为None则使用内部计数器）
            losses_dict: 包含所有损失的字典
                         必须包含: loss_start_y, loss_start_x, loss_theta, loss_length
        """
        if iteration is None:
            iteration = self._counter
            self._counter += 1
        else:
            # 更新计数器到最大迭代次数
            self._counter = max(self._counter, iteration + 1)

        # 提取分项损失
        detailed_metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss_start_y": losses_dict.get("loss_start_y", 0),
            "loss_start_x": losses_dict.get("loss_start_x", 0),
            "loss_theta": losses_dict.get("loss_theta", 0),
            "loss_length": losses_dict.get("loss_length", 0),
            # 也记录主要损失项用于对比
            "reg_xytl_loss": losses_dict.get("reg_xytl_loss", 0),
            "cls_loss": losses_dict.get("cls_loss", 0),
            "iou_loss": losses_dict.get("iou_loss", 0),
            "loss_category": losses_dict.get("loss_category", 0),
            "loss_attribute": losses_dict.get("loss_attribute", 0),
        }

        # 追加到 JSON Lines 文件
        with open(self.file_path, "a") as f:
            f.write(json.dumps(detailed_metrics) + "\n")

    def get_statistics(self):
        """
        获取损失统计信息

        Returns:
            dict: 包含各损失项的统计信息（min, max, mean, last）
        """
        if not os.path.exists(self.file_path):
            return {}

        metrics = []
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
        except Exception as e:
            logging.getLogger("unlanedet").error(f"Error reading detailed metrics: {e}")
            return {}

        if not metrics:
            return {}

        stats = {}
        loss_fields = [
            "loss_start_y",
            "loss_start_x",
            "loss_theta",
            "loss_length",
            "reg_xytl_loss",
            "cls_loss",
            "iou_loss",
            "loss_category",
            "loss_attribute",
        ]

        for field in loss_fields:
            values = [m.get(field, 0) for m in metrics]
            if values:
                stats[field] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "last": values[-1],
                    "count": len(values),
                }

        return stats

    def print_summary(self):
        """打印损失统计摘要"""
        stats = self.get_statistics()

        logger = logging.getLogger("unlanedet")
        if not stats:
            logger.info("No detailed metrics available")
            return

        logger.info("=" * 80)
        logger.info("Detailed Loss Summary")
        logger.info("=" * 80)
        logger.info(f"Records: {len(self._read_all_metrics())} iterations")

        # XYTL 分项损失
        logger.info("【XYTL Regression Loss Components】")
        header = (
            f"{'Loss':<15} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Last':>10}"
        )
        logger.info(header)
        logger.info("-" * 65)
        for field in ["loss_start_y", "loss_start_x", "loss_theta", "loss_length"]:
            if field in stats:
                s = stats[field]
                name = field.replace("loss_", "").replace("_", " ").title()
                logger.info(
                    f"{name:<15} | {s['min']:>9.4f} | {s['max']:>9.4f} | {s['mean']:>9.4f} | {s['last']:>9.4f}"
                )

        # 其他损失
        logger.info("【Other Losses】")
        logger.info(header)
        logger.info("-" * 65)
        for field in [
            "reg_xytl_loss",
            "cls_loss",
            "iou_loss",
            "loss_category",
            "loss_attribute",
        ]:
            if field in stats:
                s = stats[field]
                name = field.replace("loss_", "").replace("_", " ").title()
                logger.info(
                    f"{name:<15} | {s['min']:>9.4f} | {s['max']:>9.4f} | {s['mean']:>9.4f} | {s['last']:>9.4f}"
                )

        logger.info("=" * 80)

    def _read_all_metrics(self):
        """读取所有指标"""
        if not os.path.exists(self.file_path):
            return []

        metrics = []
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
        except Exception as e:
            logging.getLogger("unlanedet").error(f"Error reading detailed metrics: {e}")
            return []

        return metrics
