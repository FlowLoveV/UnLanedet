import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AR PL UMing CN"
INPUT_FILE = "./source/openlane_statistics/detailed_lane_parameters.npz"
OUTPUT_BASE_DIR = "./source/openlane_statistics/cluster"

BASE_CONFIG = {
    "min_ratio": 0.005,
    "feature_set": "advanced",
    "use_gpu": True,
    "max_samples": 0,
    "random_state": 42,
    "gpu_id": [4, 5, 6, 7],
}

TASKS = [
    {
        "name": "kmeans_120",
        "method": "kmeans",
        "n_clusters": 120,
        "output_suffix": "kmeans_n120",
    },
    {
        "name": "minibatch_kmeans_150_fast",
        "method": "minibatch_kmeans",
        "n_clusters": 150,
        "method_kwargs": {"batch_size": 4096, "max_iter": 200, "n_init": 3},
        "output_suffix": "minibatch_kmeans_n150",
    },
    {
        "name": "rapids_kmeans_120_gpu",
        "method": "rapids_kmeans",
        "n_clusters": 120,
        "use_gpu": True,
        "output_suffix": "rapids_kmeans_n120",
    },
    {
        "name": "gmm_120",
        "method": "gmm",
        "n_clusters": 120,
        "output_suffix": "gmm_n120",
    },
    {
        "name": "hdbscan_adv",
        "method": "hdbscan_advanced",
        "min_ratio": 0.005,
        "output_suffix": "hdbscan_adv",
    },
    {
        "name": "spectral_120",
        "method": "spectral",
        "n_clusters": 120,
        "output_suffix": "spectral_n120",
    },
    {
        "name": "hierarchical_120",
        "method": "hierarchical",
        "n_clusters": 120,
        "output_suffix": "hierarchical_n120",
    },
    {
        "name": "dbscan_basic",
        "method": "dbscan",
        "min_ratio": 0.003,
        "method_kwargs": {"eps": 0.35, "min_samples": 30},
        "output_suffix": "dbscan_basic",
    },
]
