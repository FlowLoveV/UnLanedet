import torch
import numpy as np


def init_prior_embeddings_with_stats(
    prior_embeddings, cluster_centers, stats=None, img_w=800, img_h=320
):
    """
    基于统计数据初始化先验嵌入层
    Args:
        prior_embeddings: nn.Embedding层，形状为 [num_priors, 3] (y, x, theta)
        cluster_centers: 聚类中心数组，形状为 [n_clusters, 4] (x, y, theta, length)
        stats: 统计信息字典 (可选)
        img_w: 图像宽度
        img_h: 图像高度
    """
    num_priors = prior_embeddings.weight.shape[0]
    n_clusters = len(cluster_centers)

    centers_x = cluster_centers[:, 0] / img_w
    centers_y = cluster_centers[:, 1] / img_h
    centers_theta = cluster_centers[:, 2] / 180.0

    centers_tensor = torch.stack(
        [
            torch.tensor(centers_y, dtype=torch.float32),
            torch.tensor(centers_x, dtype=torch.float32),
            torch.tensor(centers_theta, dtype=torch.float32),
        ],
        dim=1,
    )

    with torch.no_grad():
        if num_priors <= n_clusters:
            # 如果先验数小于等于聚类数，直接使用聚类中心（截断）
            prior_embeddings.weight.data = centers_tensor[:num_priors]
            print(f"先验初始化完成: 使用前 {num_priors} 个聚类中心")
        else:
            # 1. 填充聚类中心
            prior_embeddings.weight.data[:n_clusters] = centers_tensor
            # 2. 剩余的先验基于统计分布生成
            remaining = num_priors - n_clusters
            # 获取统计边界 (for reference, but we output normalized)
            # max_x = 800.0
            # max_y = 320.0
            # 起点X: 均匀分布 (Normalized 0-1)
            start_x = torch.rand(remaining)
            # 起点Y: 集中在底部 (Beta分布, Normalized 0-1)
            start_y = torch.distributions.Beta(2, 1).sample((remaining,))
            # 角度: 按统计比例生成
            left_ratio = 0.44
            if stats is not None and "thetas" in stats:
                thetas_data = stats["thetas"]
                left_ratio = np.mean(thetas_data < 0.5)
            # 生成角度
            thetas_list = []
            for i in range(remaining):
                rand_val = torch.rand(1).item()
                if rand_val < left_ratio:
                    # 左倾: 0.1-0.5
                    theta = 0.1 + 0.4 * torch.rand(1)
                else:
                    # 右倾: 0.5-0.9
                    theta = 0.5 + 0.4 * torch.rand(1)
                thetas_list.append(theta)
            thetas = torch.tensor(thetas_list)
            # 组合成先验 [start_y, start_x, theta]
            random_priors = torch.stack([start_y, start_x, thetas], dim=1)
            prior_embeddings.weight.data[n_clusters:] = random_priors
            print(
                f"先验初始化完成: {n_clusters}个聚类中心 (归一化) + {remaining}个随机先验 (归一化)"
            )
