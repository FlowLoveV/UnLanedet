import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
import argparse
import os
import os.path as osp
from tqdm import tqdm
import json
import warnings
import logging

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AR PL UMing CN"


GPU_AVAILABLE = True  # 默认假设可用，将在函数内验证

plt.rcParams["font.family"] = "AR PL UMing CN"

logger = logging.getLogger("openlane.clustering")


def load_detailed_lane_parameters(npz_path, use_gpu=False, gpu_id=None):
    """加载详细的车道线参数"""
    print(f"加载车道线参数文件: {npz_path} (GPU: {use_gpu})")

    if use_gpu:
        try:
            import cupy as cp

            if gpu_id is not None:
                cp.cuda.Device(int(gpu_id)).use()

            # 使用 numpy 加载后再转到 cupy，或者如果磁盘/网络是瓶颈，这已经足够了
            # 实际上 np.load 本身还是在 CPU 内存，之后 cp.asarray 移动到 GPU
            with np.load(npz_path) as data:
                parameters = {
                    "sample_indices": cp.asarray(data["sample_indices"]),
                    "lane_indices": cp.asarray(data["lane_indices"]),
                    "start_x": cp.asarray(data["start_x"]),
                    "start_y": cp.asarray(data["start_y"]),
                    "thetas": cp.asarray(data["thetas"]),
                    "lengths": cp.asarray(data["lengths"]),
                    "num_points": cp.asarray(data["num_points"]),
                    "total_lanes": int(data["total_lanes"]),
                }
            print(f"成功加载 {parameters['total_lanes']} 条车道线参数到 GPU")
            return parameters
        except Exception as e:
            print(f"警告: 无法将数据直接加载到 GPU ({e})，回退到 CPU")
            # 继续执行 CPU 加载

    data = np.load(npz_path)
    parameters = {
        "sample_indices": data["sample_indices"],
        "lane_indices": data["lane_indices"],
        "start_x": data["start_x"],
        "start_y": data["start_y"],
        "thetas": data["thetas"],
        "lengths": data["lengths"],
        "num_points": data["num_points"],
        "total_lanes": data["total_lanes"],
    }
    print(f"成功加载 {parameters['total_lanes']} 条车道线参数")
    return parameters


def advanced_clustering_analysis(
    parameters,
    n_clusters=100,
    min_cluster_ratio=0.005,
    method="deep_cluster",
    output_dir="./clustering_results",
    use_gpu=False,
    max_samples=0,
    method_kwargs=None,
    gpu_id=None,
):
    """
    高级聚类分析 - 使用多种先进聚类算法
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("高级车道线聚类分析")
    print(f"方法: {method}, 目标簇数: {n_clusters}")
    print(f"使用GPU: {use_gpu and GPU_AVAILABLE}")
    print("=" * 60)

    # 1. 高级特征工程
    features, feature_names, scaler = create_advanced_features(
        parameters, use_gpu=use_gpu, gpu_id=gpu_id
    )
    print(f"特征矩阵形状: {features.shape}")

    # 2. 不使用采样，直接使用全部数据
    print("使用全部数据进行聚类分析...")
    features_used = features
    parameters_used = parameters  # 使用原始参数

    # 3. 执行高级聚类
    clustering_result = apply_advanced_clustering(
        features_used,
        n_clusters,
        method,
        parameters_used,
        use_gpu,
        method_kwargs,
        gpu_id,
    )

    # 4. 过滤小簇
    if min_cluster_ratio > 0:
        clustering_result = filter_small_clusters_advanced(
            clustering_result, min_cluster_ratio
        )

    # 5. 丰富簇信息 - 使用全部数据的参数
    enriched_result = enrich_cluster_info_advanced(
        clustering_result, parameters_used, feature_names
    )

    # 6. 评估聚类质量
    clustering_scores = evaluate_clustering_quality(
        features_used, clustering_result["labels"], use_gpu=use_gpu
    )
    enriched_result.update(clustering_scores)

    # 7. 高级可视化（对大数据集进行采样以加快可视化）
    if len(features_used) > 50000:
        print("数据量过大，对可视化数据进行采样...")
        indices = np.random.choice(len(features_used), 50000, replace=False)
        features_vis = features_used[indices]
        labels_vis = clustering_result["labels"][indices]
    else:
        features_vis = features_used
        labels_vis = clustering_result["labels"]

    visualize_advanced_clustering_results(enriched_result, features_vis, output_dir)

    # 8. 降维可视化（使用采样数据）
    dimensionality_reduction_visualization(
        features_vis, labels_vis, output_dir, use_gpu=use_gpu
    )

    # 9. 保存结果
    save_advanced_clustering_results(
        enriched_result, output_dir, method, n_clusters, min_cluster_ratio
    )

    return enriched_result


def create_advanced_features(parameters, use_gpu=False, gpu_id=None):
    """创建高级特征工程"""
    print(f"进行高级特征工程 (GPU加速: {use_gpu})...")

    if use_gpu:
        try:
            import cupy as cp
            import cuml
            from cuml.preprocessing import StandardScaler as cuStandardScaler

            if gpu_id is not None:
                cp.cuda.Device(int(gpu_id)).use()

            # 数据已经在 GPU 上（如果使用 load_detailed_lane_parameters 的 GPU 路径）
            # 或者在这里转换
            start_x = (
                parameters["start_x"]
                if isinstance(parameters["start_x"], cp.ndarray)
                else cp.asarray(parameters["start_x"])
            )
            start_y = (
                parameters["start_y"]
                if isinstance(parameters["start_y"], cp.ndarray)
                else cp.asarray(parameters["start_y"])
            )
            thetas = (
                parameters["thetas"]
                if isinstance(parameters["thetas"], cp.ndarray)
                else cp.asarray(parameters["thetas"])
            )
            lengths = (
                parameters["lengths"]
                if isinstance(parameters["lengths"], cp.ndarray)
                else cp.asarray(parameters["lengths"])
            )
            xp = cp
        except Exception as e:
            print(f"警告: 无法初始化 GPU 特征工程 ({e})，回退到 CPU")
            start_x = (
                parameters["start_x"].get()
                if hasattr(parameters["start_x"], "get")
                else np.asarray(parameters["start_x"])
            )
            start_y = (
                parameters["start_y"].get()
                if hasattr(parameters["start_y"], "get")
                else np.asarray(parameters["start_y"])
            )
            thetas = (
                parameters["thetas"].get()
                if hasattr(parameters["thetas"], "get")
                else np.asarray(parameters["thetas"])
            )
            lengths = (
                parameters["lengths"].get()
                if hasattr(parameters["lengths"], "get")
                else np.asarray(parameters["lengths"])
            )
            xp = np
            use_gpu = False
    else:
        start_x = np.asarray(parameters["start_x"])
        start_y = np.asarray(parameters["start_y"])
        thetas = np.asarray(parameters["thetas"])
        lengths = np.asarray(parameters["lengths"])
        xp = np

    engineered_features = []
    feature_names = []

    # 1. 基础特征
    engineered_features.extend([start_x, start_y, thetas, xp.log1p(lengths)])
    feature_names.extend(["start_x", "start_y", "theta", "log_length"])

    # 2. 角度的高级表示
    angle_rad = thetas * 2 * xp.pi
    # 三角函数表示
    engineered_features.extend(
        [
            xp.sin(angle_rad),
            xp.cos(angle_rad),
            xp.sin(2 * angle_rad),
            xp.cos(2 * angle_rad),  # 二次谐波
        ]
    )
    feature_names.extend(["sin_theta", "cos_theta", "sin_2theta", "cos_2theta"])

    # 3. 多项式特征
    engineered_features.extend(
        [
            start_x**2,
            start_y**2,
            thetas**2,
            start_x * start_y,
            start_x * thetas,
            start_y * thetas,
        ]
    )
    feature_names.extend(
        ["x_squared", "y_squared", "theta_squared", "x_y", "x_theta", "y_theta"]
    )

    # 4. 统计特征（基于局部邻域）
    k = min(50, len(start_x) // 100)  # 邻域大小
    if k > 1:
        if use_gpu:
            try:
                from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

                coords = cp.column_stack([start_x, start_y])
                nbrs = cuNearestNeighbors(n_neighbors=k).fit(coords)
                distances, indices = nbrs.kneighbors(coords)

                # GPU 上的统计计算需要谨慎，这里使用简单的聚合
                # local_theta_std = ... (cuML 没有直接的 std 聚合)
                # 为简化且保持 GPU，我们暂时跳过复杂的局部统计，或者转回 CPU 计算这部分
                # 但为了不卡死 CPU，我们只对采样数据做这个，或者使用 cupy 向量化

                # 向量化实现 std: sqrt(mean(x^2) - mean(x)^2)
                # 这里的 indices 是 (N, k)
                neighbor_thetas = thetas[indices]  # (N, k)
                local_theta_mean = neighbor_thetas.mean(axis=1)
                local_theta_std = xp.sqrt(
                    ((neighbor_thetas - local_theta_mean[:, None]) ** 2).mean(axis=1)
                )

                neighbor_lengths = lengths[indices]
                local_length_mean = neighbor_lengths.mean(axis=1)

                engineered_features.extend(
                    [local_theta_std, xp.log1p(local_length_mean)]
                )
                feature_names.extend(["local_theta_std", "local_log_length_mean"])
            except Exception as e:
                print(f"GPU 局部特征计算失败: {e}")
        else:
            from sklearn.neighbors import NearestNeighbors

            try:
                coords = np.column_stack([start_x, start_y])
                nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
                distances, indices = nbrs.kneighbors(coords)

                # 局部统计
                local_theta_std = np.array(
                    [np.std(thetas[indices[i]]) for i in range(len(thetas))]
                )
                local_length_mean = np.array(
                    [np.mean(lengths[indices[i]]) for i in range(len(lengths))]
                )

                engineered_features.extend(
                    [local_theta_std, np.log1p(local_length_mean)]
                )
                feature_names.extend(["local_theta_std", "local_log_length_mean"])
            except:
                pass

    # 5. 分位数特征
    if use_gpu:
        engineered_features.extend(
            [
                (start_x > cp.percentile(start_x, 75)).astype(float),
                (start_y > cp.percentile(start_y, 75)).astype(float),
                (thetas > cp.percentile(thetas, 75)).astype(float),
            ]
        )
    else:
        engineered_features.extend(
            [
                (start_x > np.percentile(start_x, 75)).astype(float),
                (start_y > np.percentile(start_y, 75)).astype(float),
                (thetas > np.percentile(thetas, 75)).astype(float),
            ]
        )
    feature_names.extend(["x_high_quantile", "y_high_quantile", "theta_high_quantile"])

    # 6. 交互特征
    engineered_features.extend(
        [
            start_x * xp.sin(angle_rad),
            start_y * xp.cos(angle_rad),
            xp.log1p(lengths) * (thetas - 0.5) ** 2,
        ]
    )
    feature_names.extend(
        ["x_sin_interaction", "y_cos_interaction", "length_angle_deviation"]
    )

    # 组合所有特征
    if use_gpu:
        features = cp.column_stack(engineered_features)
        scaler = cuStandardScaler()
        features_scaled = scaler.fit_transform(features)
        # 转换回 numpy 以兼容后续逻辑（或者保持 GPU 直到聚类）
        # 为了最大化 GPU 利用率，我们返回 cupy 数组
        print(f"高级特征工程完成 (GPU): {len(feature_names)} 个特征")
        return features_scaled, feature_names, scaler
    else:
        features = np.column_stack(engineered_features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print(f"高级特征工程完成 (CPU): {len(feature_names)} 个特征")
        return features_scaled, feature_names, scaler


def downsample_features(features, max_samples=50000):
    """下采样大数据集"""
    if len(features) > max_samples:
        print(f"数据量过大 ({len(features)})，随机采样 {max_samples} 个样本...")
        indices = np.random.choice(len(features), max_samples, replace=False)
        return features[indices], indices
    else:
        return features, np.arange(len(features))


def kmeans_clustering(features, n_clusters):
    """K-means聚类"""
    from sklearn.cluster import KMeans

    print("使用K-means聚类...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
        algorithm="elkan",
        max_iter=300,
        tol=1e-4,
    )
    labels = kmeans.fit_predict(features)

    result = {
        "method": "kmeans",
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_centers": kmeans.cluster_centers_,
        "silhouette_score": silhouette_score(features, labels),
    }
    return result


def apply_advanced_clustering(
    features,
    n_clusters,
    method,
    parameters,
    use_gpu=False,
    method_kwargs=None,
    gpu_id=None,
):
    """应用高级聚类方法"""
    print(f"使用 {method} 进行高级聚类 (GPU: {use_gpu})...")

    if use_gpu:
        if method == "kmeans" or method == "rapids_kmeans":
            return rapids_kmeans(features, n_clusters, parameters, gpu_id=gpu_id)
        elif method == "gmm":
            return rapids_gmm(features, n_clusters, gpu_id=gpu_id)
        elif method in ("spectral", "spectral_advanced"):
            return rapids_spectral(features, n_clusters, gpu_id=gpu_id)
        elif method in ("hierarchical", "hierarchical_advanced"):
            return rapids_hierarchical(features, n_clusters, gpu_id=gpu_id)
        elif method == "dbscan":
            return rapids_dbscan(features, method_kwargs or {}, gpu_id=gpu_id)
        elif method == "hdbscan_advanced":
            return rapids_hdbscan(features, method_kwargs or {}, gpu_id=gpu_id)
        else:
            # 默认使用 kmeans GPU
            print(f"警告: 方法 {method} 没有显式的 GPU 实现，使用 rapids_kmeans 代替")
            return rapids_kmeans(features, n_clusters, parameters, gpu_id=gpu_id)

    # CPU 路径（仅当 use_gpu=False 时进入，但在大规模数据下应警告）
    if len(features) > 100000:
        print("警告: 大规模数据集正在使用 CPU 聚类，这可能会非常慢或导致假死")

    if method == "kmeans":
        return minibatch_kmeans_clustering(features, n_clusters, method_kwargs or {})
    elif method == "gmm":
        return gaussian_mixture_advanced(features, n_clusters, parameters)
    elif method in ("spectral", "spectral_advanced"):
        return spectral_clustering_advanced(features, n_clusters, parameters)
    elif method in ("hierarchical", "hierarchical_advanced"):
        return hierarchical_clustering_advanced(features, n_clusters, parameters)
    elif method == "dbscan":
        return dbscan_basic(features, parameters, method_kwargs or {})
    elif method == "hdbscan_advanced":
        return hdbscan_advanced(features, parameters, method_kwargs or {})
    else:
        return kmeans_clustering(features, n_clusters)


def rapids_gmm(features, n_clusters, gpu_id=None):
    """使用 RAPIDS cuML 的 GPU 加速 GMM (或回退到 KMeans 如果 GMM 不可用)"""
    print("使用 RAPIDS cuML GPU 加速 GMM...")
    try:
        import cupy as cp

        try:
            from cuml.mixture import GaussianMixture as cuGMM

            gmm_available = True
        except ImportError:
            print("警告: 当前 cuml 版本不包含 cuml.mixture，将使用 cuKMeans 代替 GMM")
            from cuml.cluster import KMeans as cuKMeans

            gmm_available = False

        if gpu_id is not None:
            cp.cuda.Device(int(gpu_id)).use()

        # 确保输入是 cupy
        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)

        if gmm_available:
            gmm = cuGMM(n_components=n_clusters, random_state=42)
            labels = gmm.fit_predict(features)
            method_name = "rapids_gmm"
        else:
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            method_name = "rapids_gmm_fallback_kmeans"

        labels_np = cp.asnumpy(labels)

        result = {
            "method": method_name,
            "labels": labels_np,
            "n_clusters": n_clusters,
            "silhouette_score": -1,  # GPU 上计算轮廓系数较慢，暂时跳过
        }
        return result
    except Exception as e:
        raise RuntimeError(f"RAPIDS cuML GMM/KMeans 失败: {e}；禁止使用 CPU") from e


def rapids_spectral(features, n_clusters, gpu_id=None):
    """使用 RAPIDS cuML 的 GPU 加速谱聚类"""
    print("使用 RAPIDS cuML GPU 加速谱聚类...")
    try:
        import cupy as cp

        try:
            from cuml.cluster import SpectralClustering as cuSpectral

            spectral_available = True
        except ImportError:
            print(
                "警告: 当前 cuml 版本不包含 cuml.cluster.SpectralClustering，将使用 cuKMeans 代替"
            )
            from cuml.cluster import KMeans as cuSpectral

            spectral_available = False

        if gpu_id is not None:
            cp.cuda.Device(int(gpu_id)).use()

        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)

        if spectral_available:
            spectral = cuSpectral(n_clusters=n_clusters, random_state=42)
            labels = spectral.fit_predict(features)
            method_name = "rapids_spectral"
        else:
            kmeans = cuSpectral(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            method_name = "rapids_spectral_fallback_kmeans"

        labels_np = cp.asnumpy(labels)

        result = {
            "method": method_name,
            "labels": labels_np,
            "n_clusters": n_clusters,
        }
        return result
    except Exception as e:
        raise RuntimeError(f"RAPIDS cuML Spectral 失败: {e}；禁止使用 CPU") from e


def rapids_hierarchical(features, n_clusters, gpu_id=None):
    """使用 RAPIDS cuML 的 GPU 加速层次聚类"""
    print("使用 RAPIDS cuML GPU 加速层次聚类...")
    try:
        import cupy as cp
        from cuml.cluster import AgglomerativeClustering as cuAgglo

        if gpu_id is not None:
            cp.cuda.Device(int(gpu_id)).use()

        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)

        agglo = cuAgglo(n_clusters=n_clusters)
        labels = agglo.fit_predict(features)

        labels_np = cp.asnumpy(labels)

        result = {
            "method": "rapids_hierarchical",
            "labels": labels_np,
            "n_clusters": n_clusters,
        }
        return result
    except Exception as e:
        raise RuntimeError(f"RAPIDS cuML Hierarchical 失败: {e}；禁止使用 CPU") from e


def rapids_hdbscan(features, kwargs, gpu_id=None):
    """使用 RAPIDS cuML 的 GPU 加速 HDBSCAN"""
    print("使用 RAPIDS cuML GPU 加速 HDBSCAN...")
    try:
        import cupy as cp
        from cuml.cluster import HDBSCAN as cuHDBSCAN

        if gpu_id is not None:
            cp.cuda.Device(int(gpu_id)).use()

        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)

        min_cluster_size = int(kwargs.get("min_cluster_size", 100))
        hdb = cuHDBSCAN(min_cluster_size=min_cluster_size)
        labels = hdb.fit_predict(features)

        labels_np = cp.asnumpy(labels)
        n_clusters = len(set(labels_np.tolist())) - (
            1 if -1 in labels_np.tolist() else 0
        )

        result = {
            "method": "rapids_hdbscan",
            "labels": labels_np,
            "n_clusters": n_clusters,
        }
        return result
    except Exception as e:
        raise RuntimeError(f"RAPIDS cuML HDBSCAN 失败: {e}；禁止使用 CPU") from e


def minibatch_kmeans_clustering(features, n_clusters, kwargs):
    """MiniBatch K-Means加速并带进度条"""
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
    import time

    batch_size = int(kwargs.get("batch_size", 4096))
    max_iter = int(kwargs.get("max_iter", 200))
    n_init = int(kwargs.get("n_init", 3))
    reassignment_ratio = float(kwargs.get("reassignment_ratio", 0.01))
    print(
        f"使用MiniBatchKMeans: n_clusters={n_clusters}, batch_size={batch_size}, max_iter={max_iter}"
    )
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=42,
    )
    start = time.time()
    n_samples = len(features)
    for i in tqdm(range(max_iter), desc="MiniBatchKMeans训练迭代", unit="iter"):
        idx = np.random.randint(0, n_samples, size=batch_size)
        mbk.partial_fit(features[idx])
    labels = mbk.predict(features)
    elapsed = time.time() - start
    print(f"MiniBatchKMeans完成，用时 {elapsed:.2f}s，样本 {n_samples}")
    result = {
        "method": "minibatch_kmeans",
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_centers": mbk.cluster_centers_,
        "silhouette_score": silhouette_score(features, labels),
        "elapsed_seconds": elapsed,
    }
    return result


def dbscan_basic(features, parameters, kwargs):
    """基本DBSCAN聚类"""
    from sklearn.cluster import DBSCAN

    eps = float(kwargs.get("eps", 0.3))
    min_samples = int(kwargs.get("min_samples", 20))
    print(f"使用DBSCAN: eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(features)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1)
    result = {
        "method": "dbscan",
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_score": (
            silhouette_score(features, labels) if n_clusters > 1 else -1
        ),
    }
    return result


def deep_clustering(features, n_clusters, parameters, use_gpu=False):
    """深度聚类 - 使用自编码器学习特征表示后进行聚类"""
    print("使用深度聚类方法...")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # 检查GPU可用性
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"使用设备: {device}")

        # 自编码器架构
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim=50):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(True),
                    nn.Linear(128, 64),
                    nn.ReLU(True),
                    nn.Linear(64, encoding_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 128),
                    nn.ReLU(True),
                    nn.Linear(128, input_dim),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        # 准备数据
        input_dim = features.shape[1]
        encoding_dim = min(50, input_dim // 2)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(features).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # 初始化模型
        model = Autoencoder(input_dim, encoding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # 训练自编码器
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch in dataloader:
                data = batch[0]
                optimizer.zero_grad()
                encoded, decoded = model(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(
                    f"自编码器训练 epoch {epoch}, loss: {total_loss/len(dataloader):.6f}"
                )

        # 提取编码特征
        model.eval()
        with torch.no_grad():
            encoded_features, _ = model(X_tensor)
            encoded_features_np = encoded_features.cpu().numpy()

        print(
            f"自编码器特征提取完成: {features.shape[1]} -> {encoded_features_np.shape[1]} 维"
        )

        # 在编码空间进行聚类（使用HDBSCAN）
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(50, len(features) // 200),
            min_samples=10,
            cluster_selection_epsilon=0.1,
        )
        labels = clusterer.fit_predict(encoded_features_np)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)

        print(f"深度聚类发现 {n_clusters} 个簇，噪声比例: {noise_ratio:.2%}")

        result = {
            "method": "deep_cluster",
            "labels": labels,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "encoding_dim": encoding_dim,
        }

        return result

    except Exception as e:
        print(f"深度聚类失败: {e}，回退到HDBSCAN")
        return hdbscan_advanced(features, parameters)


def hdbscan_advanced(features, parameters):
    """高级HDBSCAN聚类"""
    import hdbscan

    print("使用高级HDBSCAN聚类...")

    min_cluster_size = max(50, len(features) // 200)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_epsilon=0.05,
    )
    labels = clusterer.fit_predict(features)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1)

    print(f"高级HDBSCAN发现 {n_clusters} 个簇，噪声比例: {noise_ratio:.2%}")

    result = {
        "method": "hdbscan_advanced",
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
    }
    return result


def spectral_clustering_advanced(features, n_clusters, parameters):
    """高级谱聚类"""
    print("使用高级谱聚类...")

    # 使用多种核函数和参数
    best_score = -1
    best_labels = None

    # 参数网格
    param_combinations = [
        {"affinity": "rbf", "gamma": 1.0},
        {"affinity": "rbf", "gamma": 0.1},
        {"affinity": "nearest_neighbors", "n_neighbors": 10},
        {"affinity": "nearest_neighbors", "n_neighbors": 20},
    ]

    for params in param_combinations:
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                **params,
                assign_labels="cluster_qr",  # 更稳定的标签分配
            )
            labels = spectral.fit_predict(features)
            score = silhouette_score(features, labels)

            if score > best_score:
                best_score = score
                best_labels = labels
        except:
            continue

    if best_labels is None:
        # 回退到默认参数
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
        best_labels = spectral.fit_predict(features)
        best_score = silhouette_score(features, best_labels)

    result = {
        "method": "spectral_advanced",
        "labels": best_labels,
        "n_clusters": n_clusters,
        "silhouette_score": best_score,
    }

    return result


def gaussian_mixture_advanced(features, n_clusters, parameters):
    """高级高斯混合模型"""
    from sklearn.mixture import GaussianMixture

    print("使用高级高斯混合模型...")

    # 使用BIC准则选择最佳组件数和协方差类型
    n_components_range = range(max(2, n_clusters - 10), min(n_clusters + 10, 50))
    cv_types = ["spherical", "tied", "diag", "full"]

    best_bic = np.inf
    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            if n_components >= len(features):
                continue

            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                random_state=42,
                n_init=3,
            )
            gmm.fit(features)
            bic = gmm.bic(features)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=n_clusters, random_state=42)

    labels = best_gmm.fit_predict(features)
    probabilities = best_gmm.predict_proba(features)

    result = {
        "method": "gaussian_mixture_advanced",
        "labels": labels,
        "probabilities": probabilities,
        "n_clusters": best_gmm.n_components,
        "silhouette_score": silhouette_score(features, labels),
        "bic": best_gmm.bic(features),
        "aic": best_gmm.aic(features),
        "covariance_type": best_gmm.covariance_type,
    }

    return result


def hierarchical_clustering_advanced(features, n_clusters, parameters):
    """高级层次聚类"""
    print("使用高级层次聚类...")

    # 尝试不同的连接方式
    best_score = -1
    best_labels = None
    best_linkage = None

    linkage_methods = ["ward", "complete", "average", "single"]

    for linkage in linkage_methods:
        try:
            # Ward方法只能用于欧氏距离
            if linkage == "ward":
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage=linkage
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    affinity="cosine",  # 尝试余弦距离
                )

            labels = clustering.fit_predict(features)
            score = silhouette_score(features, labels)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_linkage = linkage
        except:
            continue

    result = {
        "method": "hierarchical_advanced",
        "labels": best_labels,
        "n_clusters": n_clusters,
        "silhouette_score": best_score,
        "best_linkage": best_linkage,
    }

    return result


def rapids_kmeans(features, n_clusters, parameters, gpu_id=None):
    """使用RAPIDS cuML的GPU加速K-means"""
    print("使用RAPIDS cuML GPU加速K-means...")
    try:
        import cupy as cp
        from cuml.cluster import KMeans as cuKMeans

        if gpu_id is not None:
            try:
                cp.cuda.Device(int(gpu_id)).use()
            except Exception as e:
                print(f"CUDA设备初始化警告: {e}，尝试使用默认设备")

        # 确保输入是 cupy 数组
        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)
    except Exception as e:
        print(f"CUDA设备初始化失败: {e}")

    try:
        kmeans = cuKMeans(
            n_clusters=n_clusters, max_iter=300, tol=1e-4, init="scalable-k-means++"
        )

        labels = kmeans.fit_predict(features)
        labels_np = cp.asnumpy(labels)
        cluster_centers = cp.asnumpy(kmeans.cluster_centers_)

        result = {
            "method": "rapids_kmeans",
            "labels": labels_np,
            "n_clusters": n_clusters,
            "cluster_centers": cluster_centers,
            "silhouette_score": -1,  # 同样暂时跳过 GPU 轮廓系数
        }

        return result
    except Exception as e:
        raise RuntimeError(
            f"RAPIDS cuML GPU 加速失败: {e}；禁止使用 CPU（会卡死系统）"
        ) from e


def rapids_dbscan(features, kwargs, gpu_id=None):
    """使用RAPIDS cuML的GPU加速DBSCAN"""
    print("使用RAPIDS cuML GPU加速DBSCAN...")
    try:
        import cupy as cp
        from cuml.cluster import DBSCAN as cuDBSCAN

        if gpu_id is not None:
            try:
                cp.cuda.Device(int(gpu_id)).use()
            except Exception as e:
                print(f"CUDA设备初始化警告: {e}")

        if not isinstance(features, cp.ndarray):
            features = cp.asarray(features)
    except Exception as e:
        print(f"CUDA初始化警告: {e}")

    try:
        eps = float(kwargs.get("eps", 0.3))
        min_samples = int(kwargs.get("min_samples", 20))
        db = cuDBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(features)

        labels_np = cp.asnumpy(labels).ravel()
        n_clusters = len(set(labels_np.tolist())) - (
            1 if -1 in labels_np.tolist() else 0
        )
        noise_ratio = np.mean(labels_np == -1)

        result = {
            "method": "dbscan",
            "labels": labels_np,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette_score": -1,
        }
        return result
    except Exception as e:
        raise RuntimeError(
            f"RAPIDS cuML GPU DBSCAN 失败: {e}；禁止使用 CPU（会卡死系统）"
        ) from e


def torch_kmeans_clustering(features, n_clusters, gpu_id=None):
    """使用PyTorch在GPU上进行K-means"""
    import torch

    if gpu_id is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(gpu_id))
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(features.astype(np.float32)).to(device)
    n = X.shape[0]
    if n_clusters > n:
        n_clusters = n
    indices = torch.randperm(n, device=device)[:n_clusters]
    centers = X[indices]
    for _ in range(50):
        dists = torch.cdist(X, centers)
        labels = dists.argmin(dim=1)
        new_centers = torch.stack(
            [
                X[labels == i].mean(dim=0) if (labels == i).any() else centers[i]
                for i in range(n_clusters)
            ]
        )
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    labels_np = labels.cpu().numpy()
    centers_np = centers.cpu().numpy()
    result = {
        "method": "torch_kmeans",
        "labels": labels_np,
        "n_clusters": n_clusters,
        "cluster_centers": centers_np,
        "silhouette_score": (
            silhouette_score(features, labels_np)
            if len(np.unique(labels_np)) > 1
            else -1
        ),
    }
    return result


def filter_small_clusters_advanced(clustering_result, min_ratio):
    """高级小簇过滤"""
    labels = clustering_result["labels"]
    total_points = len(labels)

    cluster_sizes = {}
    for label in set(labels):
        if label == -1:
            continue
        size = np.sum(labels == label)
        ratio = size / total_points
        cluster_sizes[label] = {"size": size, "ratio": ratio}

    valid_clusters = [
        label for label, info in cluster_sizes.items() if info["ratio"] >= min_ratio
    ]

    print(f"过滤前簇数: {len(cluster_sizes)}，过滤后簇数: {len(valid_clusters)}")

    new_labels = labels.copy()
    for label in set(labels):
        if label not in valid_clusters and label != -1:
            new_labels[labels == label] = -1

    clustering_result["labels"] = new_labels
    clustering_result["filtered_clusters"] = valid_clusters
    clustering_result["min_cluster_ratio"] = min_ratio

    return clustering_result


def enrich_cluster_info_advanced(clustering_result, parameters, feature_names):
    """丰富簇的统计信息（高级版本）"""
    labels = clustering_result["labels"]

    # 确保 labels 是 numpy 数组
    try:
        import cupy as cp

        if isinstance(labels, cp.ndarray):
            labels = cp.asnumpy(labels)
    except:
        pass

    # 确保 parameters 中的数据也是 numpy
    for key in ["start_x", "start_y", "thetas", "lengths"]:
        if key in parameters:
            try:
                import cupy as cp

                if isinstance(parameters[key], cp.ndarray):
                    parameters[key] = cp.asnumpy(parameters[key])
            except:
                pass

    # 验证参数长度与标签长度一致
    expected_length = len(labels)
    for key in ["start_x", "start_y", "thetas", "lengths"]:
        if key in parameters and len(parameters[key]) != expected_length:
            raise ValueError(
                f"参数 {key} 长度不匹配: {len(parameters[key])} != {expected_length}"
            )

    clusters_info = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        cluster_mask = labels == cluster_id
        cluster_size = np.sum(cluster_mask)

        if cluster_size == 0:
            continue

        cluster_ratio = cluster_size / len(labels)

        # 簇的详细统计信息
        cluster_stats = {
            "cluster_id": int(cluster_id),
            "size": int(cluster_size),
            "ratio": float(cluster_ratio),
            # 起点统计
            "start_x": {
                "mean": float(np.mean(parameters["start_x"][cluster_mask])),
                "std": float(np.std(parameters["start_x"][cluster_mask])),
                "min": float(np.min(parameters["start_x"][cluster_mask])),
                "max": float(np.max(parameters["start_x"][cluster_mask])),
                "q25": float(np.percentile(parameters["start_x"][cluster_mask], 25)),
                "q75": float(np.percentile(parameters["start_x"][cluster_mask], 75)),
            },
            "start_y": {
                "mean": float(np.mean(parameters["start_y"][cluster_mask])),
                "std": float(np.std(parameters["start_y"][cluster_mask])),
                "min": float(np.min(parameters["start_y"][cluster_mask])),
                "max": float(np.max(parameters["start_y"][cluster_mask])),
                "q25": float(np.percentile(parameters["start_y"][cluster_mask], 25)),
                "q75": float(np.percentile(parameters["start_y"][cluster_mask], 75)),
            },
            # 角度统计
            "theta": {
                "mean": float(np.mean(parameters["thetas"][cluster_mask])),
                "std": float(np.std(parameters["thetas"][cluster_mask])),
                "min": float(np.min(parameters["thetas"][cluster_mask])),
                "max": float(np.max(parameters["thetas"][cluster_mask])),
                "left_leaning_ratio": float(
                    np.mean(parameters["thetas"][cluster_mask] < 0.5)
                ),
                "right_leaning_ratio": float(
                    np.mean(parameters["thetas"][cluster_mask] > 0.5)
                ),
                "vertical_ratio": float(
                    np.mean(
                        (parameters["thetas"][cluster_mask] >= 0.45)
                        & (parameters["thetas"][cluster_mask] <= 0.55)
                    )
                ),
                "sharp_left_ratio": float(
                    np.mean(parameters["thetas"][cluster_mask] < 0.4)
                ),
                "sharp_right_ratio": float(
                    np.mean(parameters["thetas"][cluster_mask] > 0.6)
                ),
            },
            # 长度统计
            "length": {
                "mean": float(np.mean(parameters["lengths"][cluster_mask])),
                "std": float(np.std(parameters["lengths"][cluster_mask])),
                "min": float(np.min(parameters["lengths"][cluster_mask])),
                "max": float(np.max(parameters["lengths"][cluster_mask])),
                "mean_normalized": float(
                    np.mean(parameters["lengths"][cluster_mask]) / 1010
                ),
                "short_ratio": float(np.mean(parameters["lengths"][cluster_mask] < 50)),
                "long_ratio": float(np.mean(parameters["lengths"][cluster_mask] > 150)),
            },
            # 簇的紧凑度
            "compactness": float(
                np.mean(
                    np.linalg.norm(
                        np.column_stack(
                            [
                                parameters["start_x"][cluster_mask],
                                parameters["start_y"][cluster_mask],
                                parameters["thetas"][cluster_mask],
                                np.log1p(parameters["lengths"][cluster_mask]),
                            ]
                        )
                        - np.mean(
                            np.column_stack(
                                [
                                    parameters["start_x"][cluster_mask],
                                    parameters["start_y"][cluster_mask],
                                    parameters["thetas"][cluster_mask],
                                    np.log1p(parameters["lengths"][cluster_mask]),
                                ]
                            ),
                            axis=0,
                        ),
                        axis=1,
                    )
                )
            ),
            # 簇的代表性描述
            "description": generate_cluster_description(
                parameters["start_x"][cluster_mask],
                parameters["start_y"][cluster_mask],
                parameters["thetas"][cluster_mask],
                parameters["lengths"][cluster_mask],
            ),
        }

        clusters_info.append(cluster_stats)

    # 按簇大小排序
    clusters_info.sort(key=lambda x: x["size"], reverse=True)

    # 重新编号簇ID（按大小顺序）
    for i, cluster in enumerate(clusters_info):
        cluster["cluster_id"] = i

    # 构建完整结果
    enriched_result = {
        "clustering_method": clustering_result["method"],
        "n_clusters": len(clusters_info),
        "total_points": len(labels),
        "silhouette_score": float(clustering_result.get("silhouette_score", -1)),
        "noise_ratio": float(clustering_result.get("noise_ratio", 0)),
        "min_cluster_ratio": float(clustering_result.get("min_cluster_ratio", 0)),
        "clusters": clusters_info,
        "labels": labels.tolist(),
        "feature_names": feature_names,
    }

    # 添加方法特定的指标
    if "bic" in clustering_result:
        enriched_result["bic_score"] = float(clustering_result["bic"])
    if "aic" in clustering_result:
        enriched_result["aic_score"] = float(clustering_result["aic"])
    if "min_cluster_size" in clustering_result:
        enriched_result["min_cluster_size"] = clustering_result["min_cluster_size"]
    if "best_params" in clustering_result:
        enriched_result["best_params"] = clustering_result["best_params"]
    if "best_linkage" in clustering_result:
        enriched_result["best_linkage"] = clustering_result["best_linkage"]
    if "covariance_type" in clustering_result:
        enriched_result["covariance_type"] = clustering_result["covariance_type"]
    if "encoding_dim" in clustering_result:
        enriched_result["encoding_dim"] = clustering_result["encoding_dim"]

    return enriched_result


def generate_cluster_description(start_x, start_y, thetas, lengths):
    """生成簇的文本描述"""
    desc_parts = []

    # 位置描述
    x_mean = np.mean(start_x)
    y_mean = np.mean(start_y)

    if x_mean < 0.33:
        pos_desc = "左侧"
    elif x_mean < 0.66:
        pos_desc = "中央"
    else:
        pos_desc = "右侧"

    if y_mean < 0.33:
        pos_desc += "上部"
    elif y_mean < 0.66:
        pos_desc += "中部"
    else:
        pos_desc += "底部"

    desc_parts.append(f"{pos_desc}位置")

    # 角度描述
    theta_mean = np.mean(thetas)
    if theta_mean < 0.4:
        angle_desc = "明显左倾"
    elif theta_mean < 0.45:
        angle_desc = "轻微左倾"
    elif theta_mean < 0.55:
        angle_desc = "接近垂直"
    elif theta_mean < 0.6:
        angle_desc = "轻微右倾"
    else:
        angle_desc = "明显右倾"

    desc_parts.append(angle_desc)

    # 长度描述
    length_mean = np.mean(lengths)
    if length_mean < 50:
        length_desc = "短"
    elif length_mean < 100:
        length_desc = "中等"
    else:
        length_desc = "长"

    desc_parts.append(f"{length_desc}车道线")

    # 角度一致性
    theta_std = np.std(thetas)
    if theta_std < 0.05:
        consistency_desc = "角度一致"
    elif theta_std < 0.1:
        consistency_desc = "角度较一致"
    else:
        consistency_desc = "角度分散"

    desc_parts.append(consistency_desc)

    return ", ".join(desc_parts)


def _evaluate_gpu_scores(ftrs, lbls):
    scores = {}
    try:
        import cupy as cp
        from cuml.metrics.cluster import silhouette_score as cu_silhouette_score

        if not isinstance(ftrs, cp.ndarray):
            ftrs = cp.asarray(ftrs)
        if not isinstance(lbls, cp.ndarray):
            lbls = cp.asarray(lbls)
        n_noise_points = int(cp.sum(lbls == -1))
        total_points = int(lbls.size)
        noise_ratio = float(n_noise_points / max(1, total_points))
        valid_mask = lbls != -1
        valid_labels = lbls[valid_mask]
        n_valid_clusters = int(cp.unique(valid_labels).size)
        scores["n_valid_clusters"] = n_valid_clusters
        scores["n_noise_points"] = n_noise_points
        scores["noise_ratio"] = noise_ratio
        if n_valid_clusters >= 2:
            max_samples = 20000
            if valid_labels.size > max_samples:
                logger.info(
                    f"数据量较大 ({int(valid_labels.size)}), 采样 {max_samples} 计算轮廓系数"
                )
                idx = cp.random.choice(valid_labels.size, max_samples, replace=False)
                sample_features = ftrs[valid_mask][idx]
                sample_labels = valid_labels[idx]
            else:
                sample_features = ftrs[valid_mask]
                sample_labels = valid_labels
            sil = None
            try:
                sil = cu_silhouette_score(sample_features, sample_labels)
            except Exception as e:
                logger.warning(f"GPU 轮廓系数计算失败: {e}")
            scores["silhouette_score"] = float(sil) if sil is not None else -1.0
        else:
            scores["silhouette_score"] = -1.0
        labels_v = lbls[valid_mask]
        feats_v = ftrs[valid_mask]
        unique = cp.unique(labels_v)
        k = int(unique.size)
        if k >= 2:
            idx = cp.searchsorted(unique, labels_v)
            counts = cp.bincount(idx, minlength=k).astype(cp.float32)
            F = int(feats_v.shape[1])
            centroids = cp.zeros((k, F), dtype=feats_v.dtype)
            S = cp.zeros((k,), dtype=feats_v.dtype)
            W_total = cp.array(0.0, dtype=feats_v.dtype)
            for i in range(k):
                mask_i = idx == i
                if int(counts[i]) == 0:
                    continue
                cf = feats_v[mask_i]
                m_i = cp.mean(cf, axis=0)
                centroids[i] = m_i
                diffs = cf - m_i
                sq = cp.sum(diffs * diffs, axis=1)
                mean_sq = cp.mean(sq)
                S[i] = cp.sqrt(mean_sq)
                W_total += cp.sum(sq)
            mu = cp.mean(feats_v, axis=0)
            diffs_mu = centroids - mu
            B = cp.sum(counts * cp.sum(diffs_mu * diffs_mu, axis=1))
            N_valid = int(labels_v.size)
            if k > 1 and float(W_total) > 0:
                CH = (B / W_total) * ((N_valid - k) / (k - 1))
                scores["calinski_harabasz_score"] = float(CH)
            else:
                scores["calinski_harabasz_score"] = None
            eps = 1e-12
            diff_cent = centroids[:, None, :] - centroids[None, :, :]
            M = cp.linalg.norm(diff_cent, axis=2) + eps
            R = (S[:, None] + S[None, :]) / M
            R[cp.arange(k), cp.arange(k)] = -cp.inf
            DBI = cp.mean(cp.max(R, axis=1))
            scores["davies_bouldin_score"] = float(DBI)
        else:
            scores["calinski_harabasz_score"] = None
            scores["davies_bouldin_score"] = None
    except ImportError:
        logger.warning("无法导入 GPU 评估库，跳过 GPU 指标计算")
        scores["silhouette_score"] = -1.0
        scores["calinski_harabasz_score"] = None
        scores["davies_bouldin_score"] = None
        scores["n_valid_clusters"] = -1
        scores["n_noise_points"] = -1
        scores["noise_ratio"] = -1.0
    except Exception as e:
        logger.error(f"GPU 评估过程中出错: {e}")
        scores["silhouette_score"] = -1.0
        scores["calinski_harabasz_score"] = None
        scores["davies_bouldin_score"] = None
        scores["n_valid_clusters"] = -1
        scores["n_noise_points"] = -1
        scores["noise_ratio"] = -1.0
    return scores


def _evaluate_cpu_scores(ftrs, lbls):
    scores = {}
    try:
        import cupy as cp

        if isinstance(ftrs, cp.ndarray):
            ftrs = cp.asnumpy(ftrs)
        if isinstance(lbls, cp.ndarray):
            lbls = cp.asnumpy(lbls)
    except Exception:
        pass
    valid_mask = lbls != -1
    valid_features = ftrs[valid_mask]
    valid_labels = lbls[valid_mask]
    if len(np.unique(valid_labels)) < 2:
        logger.warning("有效簇数不足，无法进行质量评估")
        return {
            "silhouette_score": -1,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
            "n_valid_clusters": int(len(np.unique(valid_labels))),
            "n_noise_points": int(np.sum(lbls == -1)),
            "noise_ratio": float(np.mean(lbls == -1)),
        }
    try:
        sil = silhouette_score(valid_features, valid_labels)
        scores["silhouette_score"] = float(sil)
    except Exception:
        scores["silhouette_score"] = -1
    try:
        ch_score = calinski_harabasz_score(valid_features, valid_labels)
        scores["calinski_harabasz_score"] = float(ch_score)
    except Exception:
        scores["calinski_harabasz_score"] = None
    try:
        db_score = davies_bouldin_score(valid_features, valid_labels)
        scores["davies_bouldin_score"] = float(db_score)
    except Exception:
        scores["davies_bouldin_score"] = None
    scores["n_valid_clusters"] = int(len(np.unique(valid_labels)))
    scores["n_noise_points"] = int(np.sum(lbls == -1))
    scores["noise_ratio"] = float(np.mean(lbls == -1))
    return scores


def evaluate_clustering_quality(features, labels, use_gpu=False):
    """评估聚类质量"""
    logger.info("评估聚类质量...")

    scores = (
        _evaluate_gpu_scores(features, labels)
        if use_gpu
        else _evaluate_cpu_scores(features, labels)
    )
    sil_str = f"{scores.get('silhouette_score', -1):.4f}"
    ch = scores.get("calinski_harabasz_score")
    db = scores.get("davies_bouldin_score")
    ch_str = f"{ch:.2f}" if isinstance(ch, (int, float)) else "N/A"
    db_str = f"{db:.4f}" if isinstance(db, (int, float)) else "N/A"
    logger.info("聚类质量评估结果")
    logger.info(f"轮廓系数: {sil_str}")
    logger.info(f"Calinski-Harabasz指数: {ch_str}")
    logger.info(f"Davies-Bouldin指数: {db_str}")
    logger.info(f"有效簇数: {scores.get('n_valid_clusters', -1)}")
    logger.info(f"噪声点比例: {scores.get('noise_ratio', 0.0):.2%}")
    return scores


def visualize_advanced_clustering_results(enriched_result, features, output_dir):
    """高级聚类结果可视化"""
    print("生成高级可视化图表...")

    clusters = enriched_result["clusters"]
    labels = np.array(enriched_result["labels"])

    # 创建综合可视化
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    # 1. 簇大小分布
    sizes = [cluster["size"] for cluster in clusters]
    axes[0, 0].bar(range(len(sizes)), sizes, alpha=0.7, color="skyblue")
    axes[0, 0].set_xlabel("簇ID")
    axes[0, 0].set_ylabel("簇大小")
    axes[0, 0].set_title(f"簇大小分布 (共{len(clusters)}个簇)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 簇比率分布
    ratios = [cluster["ratio"] for cluster in clusters]
    axes[0, 1].bar(range(len(ratios)), ratios, alpha=0.7, color="lightgreen")
    axes[0, 1].set_xlabel("簇ID")
    axes[0, 1].set_ylabel("簇比率")
    axes[0, 1].set_title("簇比率分布")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 角度分布箱线图
    theta_means = [cluster["theta"]["mean"] for cluster in clusters]
    theta_stds = [cluster["theta"]["std"] for cluster in clusters]
    axes[0, 2].errorbar(
        range(len(theta_means)),
        theta_means,
        yerr=theta_stds,
        fmt="o",
        alpha=0.7,
        capsize=5,
        color="coral",
    )
    axes[0, 2].axhline(y=0.5, color="red", linestyle="--", label="垂直")
    axes[0, 2].set_xlabel("簇ID")
    axes[0, 2].set_ylabel("平均角度θ")
    axes[0, 2].set_title("各簇角度分布")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 起点分布散点图
    for i, cluster in enumerate(clusters[:20]):  # 只显示前20个簇
        color = plt.cm.tab20(i % 20)
        axes[1, 0].scatter(
            cluster["start_x"]["mean"],
            cluster["start_y"]["mean"],
            color=color,
            s=cluster["size"] / 10,
            alpha=0.7,
            label=f'簇{cluster["cluster_id"]}' if i < 5 else "",
        )
    axes[1, 0].set_xlabel("起点X")
    axes[1, 0].set_ylabel("起点Y")
    axes[1, 0].set_title("簇中心起点分布（大小表示簇大小）")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 角度类型分布
    left_ratios = [cluster["theta"]["left_leaning_ratio"] for cluster in clusters]
    right_ratios = [cluster["theta"]["right_leaning_ratio"] for cluster in clusters]
    vertical_ratios = [cluster["theta"]["vertical_ratio"] for cluster in clusters]

    x = range(len(clusters))
    axes[1, 1].bar(x, left_ratios, alpha=0.7, label="左倾", color="blue")
    axes[1, 1].bar(
        x, right_ratios, bottom=left_ratios, alpha=0.7, label="右倾", color="green"
    )
    axes[1, 1].bar(
        x,
        vertical_ratios,
        bottom=[left + right for left, right in zip(left_ratios, right_ratios)],
        alpha=0.7,
        label="垂直",
        color="red",
    )
    axes[1, 1].set_xlabel("簇ID")
    axes[1, 1].set_ylabel("比例")
    axes[1, 1].set_title("各簇角度类型分布")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 长度分布
    lengths = [cluster["length"]["mean"] for cluster in clusters]
    axes[1, 2].bar(range(len(lengths)), lengths, alpha=0.7, color="orange")
    axes[1, 2].set_xlabel("簇ID")
    axes[1, 2].set_ylabel("平均长度（像素）")
    axes[1, 2].set_title("各簇平均长度分布")
    axes[1, 2].grid(True, alpha=0.3)

    # 7. 起点X与角度关系
    for i, cluster in enumerate(clusters[:20]):
        color = plt.cm.tab20(i % 20)
        axes[2, 0].scatter(
            cluster["start_x"]["mean"],
            cluster["theta"]["mean"],
            color=color,
            s=cluster["size"] / 10,
            alpha=0.7,
            label=f'簇{cluster["cluster_id"]}' if i < 5 else "",
        )
    axes[2, 0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="垂直")
    axes[2, 0].set_xlabel("起点X")
    axes[2, 0].set_ylabel("角度θ")
    axes[2, 0].set_title("起点X与角度关系")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 8. 长度与角度关系
    for i, cluster in enumerate(clusters[:20]):
        color = plt.cm.tab20(i % 20)
        axes[2, 1].scatter(
            cluster["length"]["mean"],
            cluster["theta"]["mean"],
            color=color,
            s=cluster["size"] / 10,
            alpha=0.7,
        )
    axes[2, 1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="垂直")
    axes[2, 1].set_xlabel("长度（像素）")
    axes[2, 1].set_ylabel("角度θ")
    axes[2, 1].set_title("长度与角度关系")
    axes[2, 1].grid(True, alpha=0.3)

    # 9. 簇紧凑度
    compactness = [cluster["compactness"] for cluster in clusters]
    axes[2, 2].bar(range(len(compactness)), compactness, alpha=0.7, color="purple")
    axes[2, 2].set_xlabel("簇ID")
    axes[2, 2].set_ylabel("紧凑度")
    axes[2, 2].set_title("各簇紧凑度（值越小越紧凑）")
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, "advanced_clustering_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 生成簇特征雷达图
    plot_cluster_radar_charts(clusters, output_dir)


def plot_cluster_radar_charts(clusters, output_dir):
    """绘制簇特征雷达图"""
    # 选择前12个最大的簇
    n_clusters_to_plot = min(12, len(clusters))

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()

    for i in range(n_clusters_to_plot):
        cluster = clusters[i]

        # 雷达图特征
        categories = ["起点X", "起点Y", "角度θ", "长度", "左倾比", "右倾比", "紧凑度"]
        values = [
            cluster["start_x"]["mean"] * 2,  # 归一化到0-2
            cluster["start_y"]["mean"] * 2,
            cluster["theta"]["mean"] * 2,
            min(cluster["length"]["mean"] / 100, 2),  # 归一化长度
            cluster["theta"]["left_leaning_ratio"] * 2,
            cluster["theta"]["right_leaning_ratio"] * 2,
            min(cluster["compactness"] * 10, 2),  # 归一化紧凑度
        ]

        # 闭合雷达图
        values += values[:1]
        categories += categories[:1]

        angles = np.linspace(0, 2 * np.pi, len(categories)).tolist()

        ax = axes[i]
        ax.plot(angles, values, "o-", linewidth=2, label=f'簇{cluster["cluster_id"]}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=8)
        ax.set_ylim(0, 2)
        ax.set_title(f'簇{cluster["cluster_id"]} (大小:{cluster["size"]})', fontsize=10)
        ax.grid(True)

    # 隐藏多余的子图
    for i in range(n_clusters_to_plot, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, "cluster_radar_charts.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def dimensionality_reduction_visualization(features, labels, output_dir, use_gpu=False):
    """降维可视化"""
    print("进行降维可视化...")
    if use_gpu:
        try:
            import cupy as cp

            try:
                from cuml.manifold import UMAP as cuUMAP

                gpu_umap_available = True
            except Exception:
                gpu_umap_available = False

            if not gpu_umap_available:
                print("GPU 模式下跳过降维可视化：未检测到 RAPIDS cuML UMAP")
                return

            if not isinstance(features, cp.ndarray):
                features = cp.asarray(features)
            if not isinstance(labels, cp.ndarray):
                labels = cp.asarray(labels)

            if len(features) > 10000:
                indices = cp.random.choice(len(features), 10000, replace=False)
                features_sample = features[indices]
                labels_sample = labels[indices]
            else:
                features_sample = features
                labels_sample = labels

            print("使用GPU UMAP进行降维...")
            reducer = cuUMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            )
            embedding = reducer.fit_transform(features_sample)
            embedding_np = cp.asnumpy(embedding)
            labels_np = cp.asnumpy(labels_sample)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            scatter = ax.scatter(
                embedding_np[:, 0],
                embedding_np[:, 1],
                c=labels_np,
                cmap="tab20",
                alpha=0.6,
                s=1,
            )
            ax.set_title("UMAP 降维可视化 (GPU)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label="Cluster ID")
            plt.tight_layout()
            plt.savefig(
                osp.join(output_dir, "dimensionality_reduction.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return
        except Exception as e:
            print(f"GPU 降维可视化失败: {e}，跳过")
            return

    # CPU 路径：使用 t-SNE 和 UMAP（umap-learn）
    try:
        import umap
    except Exception as e:
        print(f"CPU 降维可视化依赖缺失: {e}，跳过")
        return

    # 确保是 numpy
    try:
        import cupy as cp

        if isinstance(features, cp.ndarray):
            features = cp.asnumpy(features)
        if isinstance(labels, cp.ndarray):
            labels = cp.asnumpy(labels)
    except:
        pass

    # 使用t-SNE和UMAP
    methods = [
        ("t-SNE", TSNE(n_components=2, random_state=42, perplexity=30)),
        (
            "UMAP",
            umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (method_name, reducer) in enumerate(methods):
        try:
            if len(features) > 10000:
                indices = np.random.choice(len(features), 10000, replace=False)
                features_sample = features[indices]
                labels_sample = labels[indices]
            else:
                features_sample = features
                labels_sample = labels
            print(f"使用{method_name}进行降维...")
            embedding = reducer.fit_transform(features_sample)
            scatter = axes[idx].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels_sample,
                cmap="tab20",
                alpha=0.6,
                s=1,
            )
            axes[idx].set_title(f"{method_name} 降维可视化")
            axes[idx].set_xlabel("Component 1")
            axes[idx].set_ylabel("Component 2")
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label="Cluster ID")
        except Exception as e:
            print(f"{method_name}降维失败: {e}")
            axes[idx].text(
                0.5,
                0.5,
                f"{method_name}失败",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].set_title(f"{method_name} 降维失败")

    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, "dimensionality_reduction.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_advanced_clustering_results(
    enriched_result, output_dir, method, n_clusters, min_ratio
):
    """保存高级聚类结果"""
    print("保存聚类结果...")

    # 保存为JSON格式（可读性好）
    json_path = osp.join(output_dir, f"clustering_results_{method}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_result, f, indent=2, ensure_ascii=False, default=float)

    # 保存为NPZ格式（便于程序读取）
    npz_path = osp.join(output_dir, f"clustering_results_{method}.npz")

    # 提取数组数据
    cluster_centers = []
    cluster_sizes = []
    cluster_ratios = []

    for cluster in enriched_result["clusters"]:
        cluster_centers.append(
            [
                cluster["start_x"]["mean"],
                cluster["start_y"]["mean"],
                cluster["theta"]["mean"],
                cluster["length"]["mean"],
            ]
        )
        cluster_sizes.append(cluster["size"])
        cluster_ratios.append(cluster["ratio"])

    np.savez(
        npz_path,
        cluster_centers=np.array(cluster_centers),
        cluster_sizes=np.array(cluster_sizes),
        cluster_ratios=np.array(cluster_ratios),
        labels=np.array(enriched_result["labels"]),
        clustering_method=enriched_result["clustering_method"],
        n_clusters=enriched_result["n_clusters"],
        silhouette_score=enriched_result["silhouette_score"],
        noise_ratio=enriched_result.get("noise_ratio", 0),
        min_cluster_ratio=min_ratio,
    )

    # 保存文本摘要
    summary_path = osp.join(output_dir, f"clustering_summary_{method}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("高级聚类分析结果摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"聚类方法: {enriched_result['clustering_method']}\n")
        f.write(f"总簇数: {enriched_result['n_clusters']}\n")
        f.write(f"总样本数: {enriched_result['total_points']}\n")
        f.write(f"轮廓系数: {enriched_result['silhouette_score']:.4f}\n")
        f.write(f"噪声比例: {enriched_result.get('noise_ratio', 0):.2%}\n")
        f.write(f"最小簇比率: {min_ratio:.3%}\n")

        if "calinski_harabasz_score" in enriched_result and isinstance(
            enriched_result["calinski_harabasz_score"], (int, float)
        ):
            f.write(
                f"Calinski-Harabasz指数: {enriched_result['calinski_harabasz_score']:.2f}\n"
            )
        if "davies_bouldin_score" in enriched_result and isinstance(
            enriched_result["davies_bouldin_score"], (int, float)
        ):
            f.write(
                f"Davies-Bouldin指数: {enriched_result['davies_bouldin_score']:.4f}\n"
            )

        f.write(f"\n前20个最大的簇:\n")
        f.write("-" * 100 + "\n")
        f.write(
            "簇ID |   大小   |   比率   |  起点X  |  起点Y  |  角度θ  |  长度   | 左倾比 | 右倾比 | 垂直比 | 描述\n"
        )
        f.write("-" * 100 + "\n")

        for cluster in enriched_result["clusters"][:20]:
            f.write(
                f"{cluster['cluster_id']:4d} | {cluster['size']:8d} | {cluster['ratio']:6.2%} | "
                f"{cluster['start_x']['mean']:.3f} | {cluster['start_y']['mean']:.3f} | "
                f"{cluster['theta']['mean']:.3f} | {cluster['length']['mean']:6.1f} | "
                f"{cluster['theta']['left_leaning_ratio']:.2%} | {cluster['theta']['right_leaning_ratio']:.2%} | "
                f"{cluster['theta']['vertical_ratio']:.2%} | {cluster['description'][:30]}...\n"
            )

    print(f"聚类结果已保存到:")
    print(f"  JSON文件: {json_path}")
    print(f"  NPZ文件: {npz_path}")
    print(f"  摘要文件: {summary_path}")


def main():
    """主函数：聚类分析脚本入口"""
    parser = argparse.ArgumentParser(description="高级车道线聚类分析脚本")
    parser.add_argument(
        "--input", type=str, required=True, help="输入NPZ文件路径（包含车道线参数）"
    )
    parser.add_argument(
        "--output", type=str, default="./clustering_results", help="输出目录路径"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=100, help="聚类数目（KMeans/GMM方法使用）"
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.005,
        help="最小簇比率阈值（小于此比率的簇将被过滤）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="deep_cluster",
        choices=[
            "deep_cluster",
            "hdbscan_advanced",
            "spectral_advanced",
            "gaussian_mixture_advanced",
            "hierarchical_advanced",
            "rapids_kmeans",
        ],
        help="聚类方法",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="是否使用GPU加速（如果可用）",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not osp.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        return

    # 加载车道线参数
    parameters = load_detailed_lane_parameters(args.input)

    # 执行聚类分析
    print("\n" + "=" * 60)
    print("高级车道线聚类分析")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"聚类方法: {args.method}")
    print(f"聚类数目: {args.n_clusters}")
    print(f"最小簇比率: {args.min_ratio:.3%}")
    print(f"使用GPU: {args.use_gpu}")
    print("=" * 60)

    try:
        result = advanced_clustering_analysis(
            parameters=parameters,
            n_clusters=args.n_clusters,
            min_cluster_ratio=args.min_ratio,
            method=args.method,
            output_dir=args.output,
            use_gpu=args.use_gpu,
        )

        print(f"\n聚类分析完成！")
        print(f"生成簇数: {result['n_clusters']}")
        print(f"轮廓系数: {result['silhouette_score']:.4f}")
        print(f"噪声比例: {result.get('noise_ratio', 0):.2%}")
        print(f"结果保存到: {args.output}")

    except Exception as e:
        print(f"聚类分析过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
