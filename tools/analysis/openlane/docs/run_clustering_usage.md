# run_clustering.py 使用说明与产出

## 概览
- 文件位置: tools/analysis/openlane/[run_clustering.py](file:///data1/lxy_log/workspace/ms/UnLanedet/tools/analysis/openlane/run_clustering.py)
- 功能: 批量执行聚类任务、生成报告与可视化，并将聚类结果写入综合先验 NPZ

## 前置依赖
- 输入数据（中间量）: `./source/openlane_statistics/detailed_lane_parameters.npz`
- 配置文件: tools/analysis/openlane/[clustering_config.py](file:///data1/lxy_log/workspace/ms/UnLanedet/tools/analysis/openlane/clustering_config.py)
  - 指定 INPUT_FILE、OUTPUT_BASE_DIR、BASE_CONFIG、TASKS

## 配置文件说明
- BASE_CONFIG（所有任务共享的默认参数）：
  - min_ratio、feature_set、use_gpu、max_samples、random_state
- TASKS（要执行的任务列表，每项一个聚类方法实例）：
  - 字段：
    - name：任务名
    - method：聚类方法（kmeans、minibatch_kmeans、rapids_kmeans、gmm、hdbscan_advanced、spectral、hierarchical、dbscan）
    - n_clusters：簇数（适用方法）
    - min_ratio：过滤小簇阈值
    - method_kwargs：方法特定参数（如 MiniBatchKMeans 的 batch_size/max_iter，DBSCAN 的 eps/min_samples）
    - output_suffix：输出子目录名
  - 示例见配置文件内 TASKS 列表

## 运行方式
```bash
python tools/analysis/openlane/run_clustering.py
```
- 程序将读取 clustering_config.py，顺序执行所有 TASKS
- 每个任务创建单独输出目录: `OUTPUT_BASE_DIR / output_suffix`

## 进度与加速
- MiniBatchKMeans：迭代带进度条（tqdm），显示 ETA 与总耗时
- K-Means（CPU）：使用 elkan 算法与 n_init="auto" 加速
- RAPIDS cuML：若已安装，rapids_kmeans 使用 GPU 加速

## 产出文件
- 每个任务的输出目录（例如 `./source/openlane_statistics/cluster/minibatch_kmeans_n150`）包含：
  - 聚类统计 JSON（包含簇统计、评分、噪声比例等）
  - 聚类可视化图（直方图/散点图/热力图）
  - 降维图（TSNE/UMAP/PCA）
- 批量报告：
  - `batch_analysis_report.json` 汇总所有任务结果、标注最佳结果摘要

## 与综合先验 NPZ 的集成
- 运行结束后，自动尝试将聚类结果追加写入综合先验:
  - 位置: `Path(INPUT_FILE).parent / "openlane_priors.npz"`
  - 若存在则追加写入；若不存在则跳过
- 写入字段前缀同方法名：
  - clustering_<method>_labels
  - clustering_<method>_n_clusters
  - clustering_<method>_centers（如有）
  - clustering_<method>_silhouette
  - clustering_<method>_noise_ratio（密度方法）
  - clustering_<method>_elapsed_seconds（MiniBatchKMeans）
  - clustering_<method>_probabilities（GMM）
  - clustering_<method>_best_linkage（层次聚类）

## NPZ 格式一览（综合先验）
- 文件: `openlane_priors.npz`
- 回归与 PDF（来自 openlane_analyzer.py）：
  - start_x/start_y/thetas/lengths
  - start_x_grid/start_x_pdf（500点，0.2% 间隔）
  - start_y_grid/start_y_pdf（500点，0.2% 间隔）
- 分割统计：
  - seg_positive_ratios，seg_stats_json（JSON字符串，含 mean/std/min/max/median/total_images）
- 分类统计：
  - cls_category_counts/ratios/weights
  - cls_total_samples/num_categories/most_frequent_category/least_frequent_category
  - cls_max_category_ratio/min_category_ratio/class_imbalance_ratio
- 聚类结果（由 run_clustering 追加写入）：
  - 见上文“与综合先验 NPZ 的集成”字段列表

## 常见问题
- detailed_lane_parameters.npz 与 openlane_priors.npz 的区别：
  - detailed_lane_parameters.npz 为中间量，直接用于聚类输入
  - openlane_priors.npz 为综合先验，用于模型初始化与训练，内含聚类与各种统计
- GPU 加速失败：
  - 确认已安装 RAPIDS cuML 并与 CUDA 版本匹配；否则使用 CPU K-Means/MiniBatchKMeans
