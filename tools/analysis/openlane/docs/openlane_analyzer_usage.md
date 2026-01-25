# openlane_analyzer.py 使用说明与产出

## 概览
- 文件位置: tools/analysis/openlane/[openlane_analyzer.py](file:///data1/lxy_log/workspace/ms/UnLanedet/tools/analysis/openlane/openlane_analyzer.py)
- 功能: 从 PKL 提取车道线数据，进行基础统计与可视化，生成起始坐标 PDF（500 点），并将分割/分类/回归先验与统计统一写入综合 NPZ

## 输入数据
- OpenLane PKL：由 unlanedet/data/openlane.py 生成
  - 关键字段: lanes、lane_categories、lane_vis 等

## 主要入口
- `analyze_openlane_distribution_vector_method(pkl_path, output_dir="./source/openlane_statistics")`
  - 读取 PKL，提取每条车道线的 start_x/start_y/theta/length
  - 进行基础统计与图表绘制
  - 计算起始坐标 PDF（500 点，间隔 0.2%）
  - 写入综合先验 NPZ（openlane_priors.npz）
  - 同时生成中间量 detailed_lane_parameters.npz（供聚类使用）

## 输出目录与文件
- 输出目录默认: `./source/openlane_statistics`
  - 综合先验: `openlane_priors.npz`
  - 中间量: `detailed_lane_parameters.npz`
  - 图表: 各类分布与 PDF 图（PNG）
  - 文本统计: statistics_summary.txt

## 综合先验 NPZ 内容
- 原始分布与 PDF：
  - start_x, start_y, thetas, lengths（float32）
  - start_x_grid/start_x_pdf（500 点，0.2% 间隔）
  - start_y_grid/start_y_pdf（500 点，0.2% 间隔）
- 向量法统计：
  - vector_stats_json（JSON 字符串，含 start_x/start_y/theta/length 的均值、方差、分位数、比例等）
- 分割统计：
  - seg_positive_ratios（每图像正样本比例数组）
  - seg_stats_json（JSON 字符串，含 mean/std/min/max/median/总图像数）
- 分类统计：
  - cls_category_counts/ratios/weights
  - cls_total_samples、cls_num_categories、cls_most_frequent_category、cls_least_frequent_category
  - cls_max_category_ratio、cls_min_category_ratio、cls_class_imbalance_ratio
- 聚类结果（由 run_clustering.py 追加写入）：
  - clustering_<method>_labels/n_clusters/centers/silhouette/noise_ratio/elapsed_seconds/probabilities/best_linkage

## 可视化产出
- 基础分布图：起点X/Y直方图、角度分布（标注左倾/垂直/右倾）、长度分布
- 起点分布热力图：start_x/start_y 二维密度图
- 起始坐标 PDF：X/Y 的 PDF 曲线图、二维密度图
- 分割比例直方图：segmentation_positive_ratio_distribution.png
- 分类分布与权重图：classification_category_distribution.png

## 使用示例
```bash
python tools/analysis/openlane/openlane_analyzer.py /path/to/openlane_train.pkl ./source/openlane_statistics
```
- 运行后在输出目录生成 openlane_priors.npz、detailed_lane_parameters.npz 及各类图表与文本
- 后续聚类分析请使用 run_clustering.py，该脚本会将聚类结果追加写入 openlane_priors.npz

## openlane_priors.npz 读取与使用
- 读取方式：
  - 使用 numpy.load(path, allow_pickle=True) 读取；JSON 字段以 object 数组形式保存，需要反序列化
- 常见键与用途：
  - 回归与 PDF：start_x/start_y/thetas/lengths；start_x_grid/start_x_pdf、start_y_grid/start_y_pdf 用于按先验采样
  - 分割统计：seg_positive_ratios、seg_stats_json（JSON）
  - 分类统计：cls_category_weights 直接用于损失加权，其余计数/比例用于报告
  - 聚类结果：clustering_<method>_labels/n_clusters/centers/...，按方法名前缀区分
- 示例代码：

```python
import numpy as np
import json

path = "./source/openlane_statistics/openlane_priors.npz"
data = np.load(path, allow_pickle=True)

# 基础分布与PDF
start_x = data["start_x"]
x_grid = data["start_x_grid"]; x_pdf = data["start_x_pdf"]
idx = np.random.choice(len(x_grid), size=1024, p=x_pdf / x_pdf.sum())
sampled_start_x = x_grid[idx]

# 分类权重
cls_weights = data["cls_category_weights"] if "cls_category_weights" in data.files else None

# 反序列化统计
vector_stats = json.loads(data["vector_stats_json"][0]) if "vector_stats_json" in data.files else None
seg_stats = json.loads(data["seg_stats_json"][0]) if "seg_stats_json" in data.files else None

# 读取聚类结果（示例：minibatch_kmeans）
labels_key = "clustering_minibatch_kmeans_labels"
centers_key = "clustering_minibatch_kmeans_centers"
labels = data[labels_key] if labels_key in data.files else None
centers = data[centers_key] if centers_key in data.files else None

# 根据需要集成到训练初始化/采样流程
```

- 使用建议：
  - 在数据加载阶段读取一次 NPZ，将需要的先验传入模型与损失
  - 访问键前先检查是否存在：k in data.files
  - 需要反序列化的键：vector_stats_json、seg_stats_json
  - 对 PDF 采样建议使用 np.random.choice 并归一化概率

## 注意事项
- detailed_lane_parameters.npz 为中间量，仅供聚类输入；所有统计与先验统一写入 openlane_priors.npz
- 若需要扩展更多先验（如 theta 的 PDF 或分位点），可在该脚本中计算并追加写入 openlane_priors.npz
