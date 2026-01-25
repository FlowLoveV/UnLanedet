# clustering_analyzer.py 使用说明与产出

## 概览
- 文件位置: tools/analysis/openlane/[clustering_analyzer.py](file:///data1/lxy_log/workspace/ms/UnLanedet/tools/analysis/openlane/clustering_analyzer.py)
- 功能: 对 OpenLane 车道线参数进行特征工程与多方法聚类，输出聚类评估与可视化，并可将结果写入综合先验 NPZ

## 输入数据
- 来源 NPZ: detailed_lane_parameters.npz（中间量）
  - 可通过工具函数加载: `load_detailed_lane_parameters(npz_path)` 返回字典:
    - sample_indices, lane_indices, start_x, start_y, thetas, lengths, num_points, total_lanes

## 调用入口
- 统一入口: `advanced_clustering_analysis(parameters, n_clusters, min_cluster_ratio, method, output_dir, use_gpu=False, max_samples=0, method_kwargs=None)`
- 参数说明:
  - parameters: dict，包含 start_x/start_y/thetas/lengths 等数组
  - n_clusters: 目标簇数（部分方法会自适应）
  - min_cluster_ratio: 过滤小簇的最小比例
  - method: 聚类方法（见下）
  - output_dir: 输出目录
  - use_gpu: 启用 GPU（RAPIDS cuML 可用时）
  - method_kwargs: 针对特定方法的额外参数，如 MiniBatchKMeans 的 batch_size、max_iter 等

## 支持方法
- kmeans: 标准 K-Means（使用 elkan 算法，加速收敛）
- minibatch_kmeans: MiniBatchKMeans（支持进度条，可大幅加速）
  - method_kwargs 示例: {"batch_size": 4096, "max_iter": 200, "n_init": 3}
- rapids_kmeans: GPU 加速 K-Means（RAPIDS cuML）
- gmm: 高斯混合模型（自动选择协方差类型与组件数范围，输出软概率）
- hdbscan_advanced: HDBSCAN 密度聚类（自动设置 min_cluster_size，对不规则簇鲁棒）
- spectral: 谱聚类（多核参数网格搜索）
- hierarchical: 层次聚类（多 linkage 尝试，自动选择最佳）
- dbscan: DBSCAN（eps/min_samples 可配置）

## 使用示例
```python
from clustering_analyzer import load_detailed_lane_parameters, advanced_clustering_analysis

params = load_detailed_lane_parameters("./source/openlane_statistics/detailed_lane_parameters.npz")
res = advanced_clustering_analysis(
    parameters=params,
    n_clusters=150,
    min_cluster_ratio=0.005,
    method="minibatch_kmeans",
    output_dir="./source/openlane_statistics/cluster/minibatch_kmeans_n150",
    method_kwargs={"batch_size": 4096, "max_iter": 200}
)
print(res["n_clusters"], res.get("silhouette_score"))
```

## 速度优化与进度
- K-Means: 使用 elkan 算法与 n_init="auto"，加速 CPU 场景
- MiniBatchKMeans: 部分拟合 + tqdm 迭代进度条，适合超大样本
- RAPIDS cuML: 若已安装并可用，`method="rapids_kmeans"` 可进行 GPU 加速

## 产出结果
- enriched_result 字段（核心）：
  - clustering_method, n_clusters, total_points
  - silhouette_score, noise_ratio, min_cluster_ratio
  - clusters: 每簇统计（size/ratio，start_x/start_y/theta/length 的均值/方差/分位数等）
- 可视化：
  - 聚类统计图与降维图（TSNE/UMAP/PCA）
- 序列化：
  - 保存 JSON 统计与图表至 output_dir

## 与综合先验 NPZ 的集成
- 运行器会将聚类结果追加写入综合先验: openlane_priors.npz（与 detailed_lane_parameters.npz 同级）
- 前缀命名规范：
  - clustering_<method>_labels
  - clustering_<method>_n_clusters
  - clustering_<method>_centers（如有）
  - clustering_<method>_silhouette
  - clustering_<method>_noise_ratio（密度方法）
  - clustering_<method>_elapsed_seconds（MiniBatchKMeans）
  - clustering_<method>_probabilities（GMM）
  - clustering_<method>_best_linkage（层次聚类）

## 注意事项
- detailed_lane_parameters.npz 为中间量，不与综合先验混存
- openlane_priors.npz 包含分割/分类/回归先验与聚类结果，便于训练初始化使用
