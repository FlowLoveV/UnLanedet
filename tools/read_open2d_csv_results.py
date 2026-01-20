#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D车道线评估CSV结果读取工具
读取并分析evaluate程序生成的CSV文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


class CSVResultReader:
    """CSV结果读取器"""

    # 类别ID到名称的映射
    CATEGORY_MAP = {
        0: "unknown",
        1: "white-dash",
        2: "white-solid",
        3: "double-white-dash",
        4: "double-white-solid",
        5: "white-ldash-rsolid",
        6: "white-lsolid-rdash",
        7: "yellow-dash",
        8: "yellow-solid",
        9: "double-yellow-dash",
        10: "double-yellow-solid",
        11: "yellow-ldash-rsolid",
        12: "yellow-lsolid-rdash",
        20: "left-curbside",
        21: "right-curbside"
    }

    # 属性ID到名称的映射
    ATTRIBUTE_MAP = {
        0: "unknown",
        1: "left-left",
        2: "left",
        3: "right",
        4: "right-right"
    }

    def __init__(self, csv_folder: str):
        """
        初始化CSV读取器

        Args:
            csv_folder: CSV文件所在的文件夹路径
        """
        self.csv_folder = Path(csv_folder)
        self.iou_df = None
        self.category_df = None
        self.attribute_df = None

    def load_all(self):
        """加载所有CSV文件"""
        self.load_iou_list()
        self.load_category_stats()
        self.load_attribute_stats()

    def load_iou_list(self):
        """加载IoU列表CSV"""
        iou_path = self.csv_folder / "iou_list.csv"
        if iou_path.exists():
            self.iou_df = pd.read_csv(iou_path)
            print(f"✓ 加载IoU列表: {len(self.iou_df)} 条记录")
        else:
            print(f"✗ 未找到文件: {iou_path}")

    def load_category_stats(self):
        """加载类别统计CSV"""
        category_path = self.csv_folder / "category_stats.csv"
        if category_path.exists():
            self.category_df = pd.read_csv(category_path)
            print(f"✓ 加载类别统计: {len(self.category_df)} 个类别")
        else:
            print(f"✗ 未找到文件: {category_path}")

    def load_attribute_stats(self):
        """加载属性统计CSV"""
        attribute_path = self.csv_folder / "attribute_stats.csv"
        if attribute_path.exists():
            self.attribute_df = pd.read_csv(attribute_path)
            print(f"✓ 加载属性统计: {len(self.attribute_df)} 个属性")
        else:
            print(f"✗ 未找到文件: {attribute_path}")

    def print_summary(self):
        """打印总体统计摘要"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)

        if self.category_df is not None:
            overall = self.category_df[self.category_df['CategoryID'] == 'OVERALL']
            if len(overall) > 0:
                row = overall.iloc[0]
                print(f"TP: {row['TP']}, FP: {row['FP']}, FN: {row['FN']}")
                print(f"Precision: {row['Precision']:.4f}")
                print(f"Recall: {row['Recall']:.4f}")
                print(f"F1-Score: {row['F1-Score']:.4f}")

        print("="*60)

    def print_category_stats(self, top_n: int = None):
        """
        打印类别统计

        Args:
            top_n: 仅显示前N个类别（按F1分数降序），None表示全部显示
        """
        if self.category_df is None:
            print("✗ 类别统计未加载")
            return

        print("\n" + "="*80)
        print("类别统计详情")
        print("="*80)
        print(f"{'类别ID':<10} {'类别名称':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        print("-"*80)

        df = self.category_df[self.category_df['CategoryID'] != 'OVERALL'].copy()

        # 过滤掉N/A值并转换为数值
        df['F1-Score'] = pd.to_numeric(df['F1-Score'], errors='coerce')
        df = df.sort_values('F1-Score', ascending=False)

        if top_n:
            df = df.head(top_n)

        for _, row in df.iterrows():
            cat_id = row['CategoryID']
            cat_name = row['CategoryName']
            tp = row['TP']
            fp = row['FP']
            fn = row['FN']
            precision = row['Precision']
            recall = row['Recall']
            f1 = row['F1-Score']

            p_str = f"{precision:.4f}" if precision != 'N/A' else 'N/A   '
            r_str = f"{recall:.4f}" if recall != 'N/A' else 'N/A   '
            f1_str = f"{f1:.4f}" if pd.notna(f1) else 'N/A   '

            print(f"{cat_id:<10} {cat_name:<20} {tp:>6} {fp:>6} {fn:>6} {p_str:>12} {r_str:>12} {f1_str:>12}")

        # 打印总体统计
        overall = self.category_df[self.category_df['CategoryID'] == 'OVERALL']
        if len(overall) > 0:
            print("-"*80)
            row = overall.iloc[0]
            p_str = f"{row['Precision']:.4f}" if row['Precision'] != 'N/A' else 'N/A   '
            r_str = f"{row['Recall']:.4f}" if row['Recall'] != 'N/A' else 'N/A   '
            f1_str = f"{row['F1-Score']:.4f}" if row['F1-Score'] != 'N/A' else 'N/A   '
            print(f"{'OVERALL':<10} {'Overall':<20} {row['TP']:>6} {row['FP']:>6} {row['FN']:>6} {p_str:>12} {r_str:>12} {f1_str:>12}")

        print("="*80)

    def print_attribute_stats(self, top_n: int = None):
        """
        打印属性统计

        Args:
            top_n: 仅显示前N个属性（按F1分数降序），None表示全部显示
        """
        if self.attribute_df is None:
            print("✗ 属性统计未加载")
            return

        print("\n" + "="*80)
        print("属性统计详情")
        print("="*80)
        print(f"{'属性ID':<10} {'属性名称':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        print("-"*80)

        df = self.attribute_df.copy()

        # 过滤掉N/A值并转换为数值
        df['F1-Score'] = pd.to_numeric(df['F1-Score'], errors='coerce')
        df = df.sort_values('F1-Score', ascending=False)

        if top_n:
            df = df.head(top_n)

        for _, row in df.iterrows():
            attr_id = row['AttributeID']
            attr_name = row['AttributeName']
            tp = row['TP']
            fp = row['FP']
            fn = row['FN']
            precision = row['Precision']
            recall = row['Recall']
            f1 = row['F1-Score']

            p_str = f"{precision:.4f}" if precision != 'N/A' else 'N/A   '
            r_str = f"{recall:.4f}" if recall != 'N/A' else 'N/A   '
            f1_str = f"{f1:.4f}" if pd.notna(f1) else 'N/A   '

            print(f"{attr_id:<10} {attr_name:<20} {tp:>6} {fp:>6} {fn:>6} {p_str:>12} {r_str:>12} {f1_str:>12}")

        print("="*80)

    def print_iou_summary(self, bins: List[float] = None):
        """
        打印IoU分布统计

        Args:
            bins: 自定义IoU分箱边界，例如[0, 0.5, 0.7, 0.9, 1.0]
        """
        if self.iou_df is None:
            print("✗ IoU列表未加载")
            return

        print("\n" + "="*60)
        print("IoU分布统计")
        print("="*60)

        iou_values = self.iou_df['IoU'].values

        print(f"IoU记录总数: {len(iou_values)}")
        print(f"平均IoU: {np.mean(iou_values):.4f}")
        print(f"中位数IoU: {np.median(iou_values):.4f}")
        print(f"最小IoU: {np.min(iou_values):.4f}")
        print(f"最大IoU: {np.max(iou_values):.4f}")
        print(f"标准差: {np.std(iou_values):.4f}")

        # 分箱统计
        if bins is None:
            bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        counts, _ = np.histogram(iou_values, bins=bins)
        print("\nIoU区间分布:")
        for i in range(len(bins) - 1):
            bin_range = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
            percentage = (counts[i] / len(iou_values)) * 100 if len(iou_values) > 0 else 0
            print(f"  {bin_range}: {counts[i]:4d} ({percentage:5.2f}%)")

        print("="*60)

    def get_worst_matches(self, n: int = 10):
        """
        获取IoU最低的N个匹配

        Args:
            n: 返回的数量
        """
        if self.iou_df is None:
            print("✗ IoU列表未加载")
            return None

        worst = self.iou_df.nsmallest(n, 'IoU')
        print(f"\n最差的 {n} 个匹配:")
        print(f"{'图片名称':<60} {'IoU':>10} {'标注类别':>12} {'检测类别':>12}")
        print("-"*96)
        for _, row in worst.iterrows():
            cat_anno = self.CATEGORY_MAP.get(row['AnnoCategory'], 'unknown')
            cat_detect = self.CATEGORY_MAP.get(row['DetectCategory'], 'unknown')
            print(f"{row['ImageName']:<60} {row['IoU']:>10.4f} {cat_anno:>12} {cat_detect:>12}")

        return worst

    def get_category_wise_stats(self) -> pd.DataFrame:
        """
        获取按类别统计的详细信息

        Returns:
            包含类别统计的DataFrame
        """
        if self.category_df is None:
            return None

        return self.category_df[self.category_df['CategoryID'] != 'OVERALL']

    def get_attribute_wise_stats(self) -> pd.DataFrame:
        """
        获取按属性统计的详细信息

        Returns:
            包含属性统计的DataFrame
        """
        if self.attribute_df is None:
            return None

        return self.attribute_df

    def export_to_excel(self, output_path: str):
        """
        将三个CSV文件导出为一个Excel文件的三个Sheet

        Args:
            output_path: 输出的Excel文件路径
        """
        if self.iou_df is None or self.category_df is None or self.attribute_df is None:
            print("✗ 部分数据未加载，无法导出")
            return

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.iou_df.to_excel(writer, sheet_name='IoU_List', index=False)
            self.category_df.to_excel(writer, sheet_name='Category_Stats', index=False)
            self.attribute_df.to_excel(writer, sheet_name='Attribute_Stats', index=False)

        print(f"✓ 已导出到Excel: {output_path}")

    def plot_category_performance(self, save_path: str = None):
        """
        绘制类别性能对比图

        Args:
            save_path: 保存路径，None表示不保存
        """
        if self.category_df is None:
            print("✗ 类别统计未加载")
            return

        df = self.category_df[self.category_df['CategoryID'] != 'OVERALL'].copy()
        df['F1-Score'] = pd.to_numeric(df['F1-Score'], errors='coerce')
        df = df.sort_values('F1-Score', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(df))
        f1_scores = df['F1-Score'].values
        category_names = df['CategoryName'].values

        bars = ax.barh(y_pos, f1_scores, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(category_names)
        ax.invert_yaxis()
        ax.set_xlabel('F1-Score')
        ax.set_title('各类别检测性能对比 (F1-Score)')
        ax.set_xlim([0, 1.05])

        # 在条形上显示数值
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            if pd.notna(score):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """示例用法"""
    import argparse

    parser = argparse.ArgumentParser(description='读取2D车道线评估CSV结果')
    parser.add_argument('--csv-folder', type=str, default='./csv_results/',
                        help='CSV文件所在文件夹路径')
    parser.add_argument('--top-n', type=int, default=None,
                        help='仅显示前N个类别/属性（按F1分数降序）')
    parser.add_argument('--worst-n', type=int, default=10,
                        help='显示最差的N个匹配')
    parser.add_argument('--excel', type=str, default=None,
                        help='导出为Excel文件的路径')
    parser.add_argument('--plot', type=str, default=None,
                        help='保存类别性能对比图的路径')

    args = parser.parse_args()

    # 创建读取器
    reader = CSVResultReader(args.csv_folder)

    # 加载所有CSV文件
    reader.load_all()

    # 打印摘要
    reader.print_summary()

    # 打印类别统计
    reader.print_category_stats(top_n=args.top_n)

    # 打印属性统计
    reader.print_attribute_stats(top_n=args.top_n)

    # 打印IoU分布
    reader.print_iou_summary()

    # 显示最差的匹配
    reader.get_worst_matches(n=args.worst_n)

    # 导出到Excel
    if args.excel:
        reader.export_to_excel(args.excel)

    # 绘制性能对比图
    if args.plot:
        reader.plot_category_performance(save_path=args.plot)


if __name__ == '__main__':
    main()
