import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import multiprocessing

# 限制 CPU 线程数，防止 CPU 占用过高 (控制在 2500% 以内)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# 设置多进程启动模式为 spawn，以避免 CUDA 初始化冲突
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

plt.rcParams["font.family"] = "AR PL UMing CN"

from clustering_analyzer import (
    advanced_clustering_analysis,
    load_detailed_lane_parameters,
)


def run_config_worker(input_file, output_base_dir, config):
    try:
        # 确保 gpu_id 是单个整数，而不是列表
        gpu_id = config.get("gpu_id")
        if isinstance(gpu_id, (list, tuple)):
            if len(gpu_id) > 0:
                gpu_id = gpu_id[0]
            else:
                gpu_id = None

        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # 在设置了 CUDA_VISIBLE_DEVICES 后，逻辑设备 ID 应该是 0
            config["gpu_id"] = 0

        # 加载车道线参数，支持直接加载到 GPU
        parameters = load_detailed_lane_parameters(
            input_file,
            use_gpu=config.get("use_gpu", False),
            gpu_id=config.get("gpu_id"),
        )
        result = advanced_clustering_analysis(
            parameters=parameters,
            n_clusters=config.get("n_clusters", 100),
            min_cluster_ratio=config["min_ratio"],
            method=config["method"],
            output_dir=str(
                Path(output_base_dir)
                / (
                    config.get("output_suffix")
                    or f"{config['method']}_n{config.get('n_clusters','auto')}"
                )
            ),
            use_gpu=config.get("use_gpu", False),
            method_kwargs=config.get("method_kwargs"),
            gpu_id=config.get("gpu_id"),
        )
        priors_path = Path(input_file).parent / "openlane_priors.npz"
        if priors_path.exists():
            base = _load_npz_as_dict(str(priors_path))
            extra = _result_to_npz_fields(result)
            base.update(extra)
            _save_npz_from_dict(str(priors_path), base)
        else:
            base = _result_to_npz_fields(result)
            _save_npz_from_dict(str(priors_path), base)
        return (config.get("name"), True, result, None)
    except Exception as e:
        return (config.get("name"), False, None, str(e))


try:
    from clustering_config import INPUT_FILE as CFG_INPUT_FILE
    from clustering_config import OUTPUT_BASE_DIR as CFG_OUTPUT_BASE_DIR
    from clustering_config import BASE_CONFIG as CFG_BASE_CONFIG
    from clustering_config import TASKS as CFG_TASKS

    CONFIG_AVAILABLE = True
except Exception:
    CONFIG_AVAILABLE = False


class OpenLaneClusteringRunner:
    """OpenLane车道线聚类分析运行器"""

    def __init__(self, input_file, output_base_dir="./clustering_results"):
        """
        初始化聚类分析运行器

        Args:
            input_file: 输入NPZ文件路径
            output_base_dir: 输出基础目录
        """
        self.input_file = input_file
        self.output_base_dir = Path(output_base_dir)
        self.results = {}

        # 验证输入文件
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

    def _append_clustering_to_priors(self, priors_path, result):
        """将聚类结果追加到综合先验NPZ"""
        import numpy as np

        # 读取已有NPZ
        try:
            data = np.load(str(priors_path), allow_pickle=True)
            base = {k: data[k] for k in data.files}
        except Exception:
            base = {}
        # 将结果转为字段
        method = result.get("method", "unknown")
        prefix = f"clustering_{method}"

        def as_np_array(x):
            if x is None:
                return None
            try:
                return np.array(x)
            except Exception:
                return np.array([x], dtype=object)

        fields = {}
        labels = result.get("labels")
        n_clusters = result.get("n_clusters")
        centers = result.get("cluster_centers")
        sil = result.get("silhouette_score")
        noise = result.get("noise_ratio")
        elapsed = result.get("elapsed_seconds")
        probs = result.get("probabilities")
        best_linkage = result.get("best_linkage")
        if labels is not None:
            fields[f"{prefix}_labels"] = as_np_array(labels)
        if n_clusters is not None:
            fields[f"{prefix}_n_clusters"] = as_np_array([n_clusters])
        if centers is not None:
            fields[f"{prefix}_centers"] = as_np_array(centers)
        if sil is not None:
            fields[f"{prefix}_silhouette"] = as_np_array([sil])
        if noise is not None:
            fields[f"{prefix}_noise_ratio"] = as_np_array([noise])
        if elapsed is not None:
            fields[f"{prefix}_elapsed_seconds"] = as_np_array([elapsed])
        if probs is not None:
            fields[f"{prefix}_probabilities"] = as_np_array(probs)
        if best_linkage is not None:
            fields[f"{prefix}_best_linkage"] = as_np_array([best_linkage])
        base.update(fields)
        np.savez(str(priors_path), **base)

    def run_single_analysis(self, config):
        """
        运行单次聚类分析

        Args:
            config: 分析配置字典，包含以下键：
                - method: 聚类方法
                - n_clusters: 聚类数目
                - min_ratio: 最小簇比率
                - feature_set: 特征集
                - use_gpu: 是否使用GPU
                - max_samples: 最大采样数
                - random_state: 随机种子
                - output_suffix: 输出目录后缀
        """
        # 创建输出目录
        output_dir = self._create_output_directory(config)

        print(f"\n{'='*60}")
        print(f"开始聚类分析: {config.get('name', '未命名分析')}")
        print(f"{'='*60}")
        print(f"输入文件: {self.input_file}")
        print(f"聚类方法: {config['method']}")
        print(f"聚类数目: {config.get('n_clusters', '自动')}")
        print(f"最小比率: {config['min_ratio']}")
        print(f"特征集: {config['feature_set']}")
        print(f"使用GPU: {config['use_gpu']}")
        if "gpu_id" in config:
            print(f"GPU编号: {config['gpu_id']}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}")

        try:
            # 规范化并设置 GPU 环境
            gpu_id = config.get("gpu_id")
            if isinstance(gpu_id, (list, tuple)):
                gpu_id = gpu_id[0] if len(gpu_id) > 0 else None
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                config["gpu_id"] = 0

            # 加载车道线参数，支持直接加载到 GPU
            parameters = load_detailed_lane_parameters(
                self.input_file,
                use_gpu=config.get("use_gpu", False),
                gpu_id=config.get("gpu_id"),
            )

            # 执行聚类分析
            result = advanced_clustering_analysis(
                parameters=parameters,
                n_clusters=config.get("n_clusters", 100),
                min_cluster_ratio=config["min_ratio"],
                method=config["method"],
                output_dir=output_dir,
                use_gpu=config["use_gpu"],
                method_kwargs=config.get("method_kwargs"),
                gpu_id=config.get("gpu_id"),
            )

            # 将聚类结果写入与先验同目录的综合NPZ
            try:
                priors_path = Path(self.input_file).parent / "openlane_priors.npz"
                if priors_path.exists():
                    self._append_clustering_to_priors(priors_path, result)
                    print(f"已将聚类结果写入综合先验NPZ: {priors_path}")
                else:
                    print(f"未找到综合先验NPZ: {priors_path}，跳过写入")
            except Exception as e:
                print(f"写入聚类结果到综合NPZ失败: {e}")

            # 记录结果
            analysis_key = config.get(
                "name", f"{config['method']}_{config.get('n_clusters', 'auto')}"
            )
            self.results[analysis_key] = {
                "config": config.copy(),
                "result": result,
                "output_dir": output_dir,
                "success": True,
            }

            # 输出结果摘要
            print(f"\n✓ 聚类分析完成!")
            print(f"生成簇数: {result['n_clusters']}")
            print(f"轮廓系数: {result.get('silhouette_score', 'N/A'):.4f}")
            print(f"噪声比例: {result.get('noise_ratio', 0):.2%}")
            print(f"结果保存到: {output_dir}")

            return result

        except Exception as e:
            print(f"✗ 聚类分析失败: {e}")
            import traceback

            traceback.print_exc()

            analysis_key = config.get(
                "name", f"{config['method']}_{config.get('n_clusters', 'auto')}"
            )
            self.results[analysis_key] = {
                "config": config.copy(),
                "error": str(e),
                "success": False,
            }
            return None

    def run_batch_analysis(self, configs):
        """
        运行批量聚类分析

        Args:
            configs: 配置字典列表
        """
        print(f"开始批量聚类分析，共 {len(configs)} 个配置")

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] 运行配置: {config.get('name', f'配置{i}')}")
            self.run_single_analysis(config)

        # 生成批量分析报告
        self._generate_batch_report()

    def run_method_comparison(self, methods, base_config):
        """
        运行方法比较分析

        Args:
            methods: 要比较的方法列表
            base_config: 基础配置
        """
        configs = []
        for method in methods:
            config = base_config.copy()
            config["method"] = method
            config["name"] = f"method_{method}"
            configs.append(config)

        self.run_batch_analysis(configs)

        # 生成方法比较报告
        self._generate_method_comparison_report(methods)

    def run_parameter_sweep(self, method, param_name, param_values, base_config):
        """
        运行参数扫描分析

        Args:
            method: 聚类方法
            param_name: 参数名称 ('n_clusters', 'min_ratio', 等)
            param_values: 参数值列表
            base_config: 基础配置
        """
        configs = []
        for value in param_values:
            config = base_config.copy()
            config["method"] = method
            config[param_name] = value
            config["name"] = f"{method}_{param_name}_{value}"
            configs.append(config)

        self.run_batch_analysis(configs)

        # 生成参数扫描报告
        self._generate_parameter_sweep_report(method, param_name, param_values)

    def _create_output_directory(self, config):
        """创建输出目录"""
        if "output_suffix" in config:
            dir_name = config["output_suffix"]
        else:
            method = config["method"]
            n_clusters = config.get("n_clusters", "auto")
            dir_name = f"{method}_n{n_clusters}"

        output_dir = self.output_base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def _generate_batch_report(self):
        """生成批量分析报告"""
        report_path = self.output_base_dir / "batch_analysis_report.json"

        # 准备报告数据
        report_data = {
            "input_file": self.input_file,
            "total_analyses": len(self.results),
            "successful_analyses": sum(
                1 for r in self.results.values() if r["success"]
            ),
            "failed_analyses": sum(
                1 for r in self.results.values() if not r["success"]
            ),
            "analyses": {},
        }

        # 添加每个分析的结果
        for name, result_info in self.results.items():
            if result_info["success"]:
                result = result_info["result"]
                report_data["analyses"][name] = {
                    "config": result_info["config"],
                    "n_clusters": result["n_clusters"],
                    "silhouette_score": result.get("silhouette_score", -1),
                    "noise_ratio": result.get("noise_ratio", 0),
                    "output_dir": result_info["output_dir"],
                    "success": True,
                }
            else:
                report_data["analyses"][name] = {
                    "config": result_info["config"],
                    "error": result_info["error"],
                    "success": False,
                }

        # 找到最佳结果
        successful_results = [
            r for r in report_data["analyses"].values() if r["success"]
        ]
        if successful_results:
            best_result = max(
                successful_results, key=lambda x: x.get("silhouette_score", -1)
            )
            report_data["best_result"] = best_result

        # 保存报告
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n批量分析报告已保存到: {report_path}")

        # 打印摘要
        self._print_batch_summary(report_data)

        return report_data

    def _generate_method_comparison_report(self, methods):
        """生成方法比较报告"""
        # 比较不同方法的性能
        successful_results = {
            name: info
            for name, info in self.results.items()
            if info["success"] and info["config"]["method"] in methods
        }

        if not successful_results:
            print("没有成功的结果可用于方法比较")
            return

        print(f"\n{'='*60}")
        print("方法比较结果")
        print(f"{'='*60}")
        print(f"{'方法':<15} {'簇数':<8} {'轮廓系数':<10} {'噪声比例':<12}")
        print(f"{'-'*60}")

        for method in methods:
            method_results = [
                info
                for info in successful_results.values()
                if info["config"]["method"] == method
            ]
            if method_results:
                # 取第一个结果（假设每个方法只运行一次）
                result = method_results[0]["result"]
                print(
                    f"{method:<15} {result['n_clusters']:<8} {result.get('silhouette_score', -1):<10.4f} {result.get('noise_ratio', 0):<12.2%}"
                )

    def _generate_parameter_sweep_report(self, method, param_name, param_values):
        """生成参数扫描报告"""
        # 分析参数对性能的影响
        successful_results = {
            name: info
            for name, info in self.results.items()
            if info["success"] and info["config"]["method"] == method
        }

        if not successful_results:
            print("没有成功的结果可用于参数扫描分析")
            return

        # 提取参数和性能指标
        param_performance = []
        for value in param_values:
            result_key = f"{method}_{param_name}_{value}"
            if result_key in successful_results:
                result = successful_results[result_key]["result"]
                param_performance.append(
                    {
                        "parameter_value": value,
                        "n_clusters": result["n_clusters"],
                        "silhouette_score": result.get("silhouette_score", -1),
                        "noise_ratio": result.get("noise_ratio", 0),
                    }
                )

        if param_performance:
            print(f"\n{'='*60}")
            print(f"{method} 方法 {param_name} 参数扫描结果")
            print(f"{'='*60}")
            print(f"{param_name:<12} {'簇数':<8} {'轮廓系数':<10} {'噪声比例':<12}")
            print(f"{'-'*60}")

            for perf in sorted(param_performance, key=lambda x: x["parameter_value"]):
                print(
                    f"{perf['parameter_value']:<12} {perf['n_clusters']:<8} {perf['silhouette_score']:<10.4f} {perf['noise_ratio']:<12.2%}"
                )

            # 可视化参数扫描结果（可选）
            self._plot_parameter_sweep(param_performance, method, param_name)

    def _plot_parameter_sweep(self, param_performance, method, param_name):
        """绘制参数扫描结果图"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 轮廓系数 vs 参数
            param_values = [p["parameter_value"] for p in param_performance]
            silhouette_scores = [p["silhouette_score"] for p in param_performance]

            ax1.plot(param_values, silhouette_scores, "o-", linewidth=2, markersize=8)
            ax1.set_xlabel(param_name)
            ax1.set_ylabel("轮廓系数")
            ax1.set_title(f"{method}方法 - {param_name}参数扫描")
            ax1.grid(True, alpha=0.3)

            # 找到最佳参数
            best_idx = np.argmax(silhouette_scores)
            ax1.plot(
                param_values[best_idx], silhouette_scores[best_idx], "ro", markersize=10
            )
            ax1.annotate(
                f"最佳: {param_values[best_idx]}",
                xy=(param_values[best_idx], silhouette_scores[best_idx]),
                xytext=(10, 10),
                textcoords="offset points",
            )

            # 簇数 vs 参数
            n_clusters = [p["n_clusters"] for p in param_performance]
            ax2.plot(
                param_values,
                n_clusters,
                "s-",
                linewidth=2,
                markersize=8,
                color="orange",
            )
            ax2.set_xlabel(param_name)
            ax2.set_ylabel("簇数")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图像
            plot_path = (
                self.output_base_dir / f"parameter_sweep_{method}_{param_name}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"参数扫描图已保存到: {plot_path}")

        except Exception as e:
            print(f"绘制参数扫描图时出错: {e}")

    def _print_batch_summary(self, report_data):
        """打印批量分析摘要"""
        print(f"\n{'='*60}")
        print("批量分析摘要")
        print(f"{'='*60}")
        print(f"总分析数: {report_data['total_analyses']}")
        print(f"成功: {report_data['successful_analyses']}")
        print(f"失败: {report_data['failed_analyses']}")

        if "best_result" in report_data:
            best = report_data["best_result"]
            print(f"\n最佳结果:")
            print(f"  方法: {best['config']['method']}")
            print(f"  聚类数: {best.get('n_clusters', 'N/A')}")
            print(f"  轮廓系数: {best.get('silhouette_score', 'N/A'):.4f}")
            print(f"  输出目录: {best.get('output_dir', 'N/A')}")


# ============================================================================
# 使用示例和配置
# ============================================================================


def main():
    """主函数 - 直接运行聚类分析"""

    if CONFIG_AVAILABLE:
        INPUT_FILE = CFG_INPUT_FILE
        OUTPUT_BASE_DIR = CFG_OUTPUT_BASE_DIR
        BASE_CONFIG = CFG_BASE_CONFIG
        TASKS = CFG_TASKS
        gpu_env = os.environ.get("OPENLANE_GPU_ID")
        if gpu_env is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_env)
        runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_BASE_DIR)
        configs = []
        for task in TASKS:
            cfg = BASE_CONFIG.copy()
            cfg["method"] = task["method"]
            cfg["name"] = task.get("name", task["method"])
            if "n_clusters" in task:
                cfg["n_clusters"] = task["n_clusters"]
            if "use_gpu" in task:
                cfg["use_gpu"] = task["use_gpu"]
            if "gpu_id" in task:
                cfg["gpu_id"] = task["gpu_id"]
            elif gpu_env is not None:
                cfg["gpu_id"] = int(gpu_env)
            elif "gpu_id" in BASE_CONFIG and isinstance(BASE_CONFIG["gpu_id"], int):
                cfg["gpu_id"] = BASE_CONFIG["gpu_id"]
            if "min_ratio" in task:
                cfg["min_ratio"] = task["min_ratio"]
            if "output_suffix" in task:
                cfg["output_suffix"] = task["output_suffix"]
            if "method_kwargs" in task:
                cfg["method_kwargs"] = task["method_kwargs"]
            configs.append(cfg)
        print(f"从配置文件读取 {len(configs)} 个聚类任务")
        # 并行GPU池支持
        gpu_ids_env = os.environ.get("OPENLANE_GPU_IDS")
        try:
            from clustering_config import GPU_IDS as CFG_GPU_IDS

            GPU_IDS = CFG_GPU_IDS
        except Exception:
            GPU_IDS = None
        if (
            GPU_IDS is None
            and "gpu_id" in BASE_CONFIG
            and isinstance(BASE_CONFIG["gpu_id"], (list, tuple))
        ):
            GPU_IDS = list(BASE_CONFIG["gpu_id"])
        if gpu_ids_env and not GPU_IDS:
            try:
                GPU_IDS = [int(x) for x in gpu_ids_env.split(",") if x.strip() != ""]
            except Exception:
                GPU_IDS = None
        if GPU_IDS and len(GPU_IDS) > 0:
            print(f"检测到多GPU配置: {GPU_IDS}，启用并行执行")
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # 为每个任务分配GPU并并行执行
            for i, cfg in enumerate(configs):
                # 强制覆盖为单个 GPU ID
                cfg["gpu_id"] = GPU_IDS[i % len(GPU_IDS)]
            with ProcessPoolExecutor(max_workers=len(GPU_IDS)) as ex:
                futures = [
                    ex.submit(run_config_worker, INPUT_FILE, OUTPUT_BASE_DIR, cfg)
                    for cfg in configs
                ]
                for fut in as_completed(futures):
                    name, ok, result, err = fut.result()
                    if ok:
                        print(
                            f"✓ 并行任务完成: {name}, 簇数={result['n_clusters']}, 轮廓系数={result.get('silhouette_score', -1):.4f}"
                        )
                    else:
                        print(f"✗ 并行任务失败: {name}, 错误: {err}")
                        print("检测到错误，正在停止所有任务并退出程序...")
                        sys.exit(1)
        else:
            # 顺序执行
            for i, config in enumerate(configs, 1):
                print(f"\n[{i}/{len(configs)}] 运行配置: {config.get('name')}")
                parameters = load_detailed_lane_parameters(INPUT_FILE)
                result = advanced_clustering_analysis(
                    parameters=parameters,
                    n_clusters=config.get("n_clusters", 100),
                    min_cluster_ratio=config["min_ratio"],
                    method=config["method"],
                    output_dir=str(
                        Path(OUTPUT_BASE_DIR)
                        / (
                            config.get("output_suffix")
                            or f"{config['method']}_n{config.get('n_clusters','auto')}"
                        )
                    ),
                    use_gpu=config.get("use_gpu", False),
                    method_kwargs=config.get("method_kwargs"),
                    gpu_id=config.get("gpu_id"),
                )
                try:
                    priors_path = Path(INPUT_FILE).parent / "openlane_priors.npz"
                    if priors_path.exists():
                        runner._append_clustering_to_priors(priors_path, result)
                    else:
                        base = {}
                        extra = _result_to_npz_fields(result)
                        base.update(extra)
                        _save_npz_from_dict(str(priors_path), base)
                    print(f"已将聚类结果写入综合先验NPZ: {priors_path}")
                except Exception as e:
                    print(f"写入聚类结果到综合NPZ失败: {e}")

    # ==================== 分析场景选择 ====================
    # 取消注释要运行的场景

    SCENARIO = (
        "single"  # 可选: "single", "method_comparison", "parameter_sweep", "batch"
    )

    # ==================== 场景1: 单次分析 ====================
    if SCENARIO == "single":
        single_config = {
            "name": "my_single_analysis",  # 分析名称
            "method": "kmeans",  # 聚类方法
            "n_clusters": 96,  # 聚类数目
            **BASE_CONFIG,
        }

        runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_BASE_DIR)
        runner.run_single_analysis(single_config)

    # ==================== 场景2: 方法比较 ====================
    elif SCENARIO == "method_comparison":
        methods_to_compare = ["kmeans", "gmm", "dbscan"]

        runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_BASE_DIR)
        runner.run_method_comparison(methods_to_compare, BASE_CONFIG)

    # ==================== 场景3: 参数扫描 ====================
    elif SCENARIO == "parameter_sweep":
        method = "kmeans"
        parameter = "n_clusters"  # 要扫描的参数
        parameter_values = [50, 100, 150, 200, 250]  # 参数值列表

        runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_BASE_DIR)
        runner.run_parameter_sweep(method, parameter, parameter_values, BASE_CONFIG)

    # ==================== 场景4: 批量分析 ====================
    elif SCENARIO == "batch":
        batch_configs = [
            {
                "name": "kmeans_basic",
                "method": "kmeans",
                "n_clusters": 100,
                "min_ratio": 0.01,
                "feature_set": "basic",
            },
            {
                "name": "kmeans_advanced",
                "method": "kmeans",
                "n_clusters": 150,
                "min_ratio": 0.005,
                "feature_set": "advanced",
            },
            {
                "name": "gmm_standard",
                "method": "gmm",
                "n_clusters": 100,
                "min_ratio": 0.005,
            },
            {"name": "dbscan_density", "method": "dbscan", "min_ratio": 0.002},
        ]

        # 合并基础配置
        for config in batch_configs:
            config.update({k: v for k, v in BASE_CONFIG.items() if k not in config})

        runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_BASE_DIR)
        runner.run_batch_analysis(batch_configs)

    else:
        print(f"未知场景: {SCENARIO}")
        print("可用场景: single, method_comparison, parameter_sweep, batch")


# ============================================================================
# 高级使用示例
# ============================================================================


def advanced_example():
    """高级使用示例"""

    # 配置
    INPUT_FILE = "./source/openlane_statistics/detailed_lane_parameters.npz"
    OUTPUT_DIR = "./source/openlane_statistics/cluster"

    # 创建运行器
    runner = OpenLaneClusteringRunner(INPUT_FILE, OUTPUT_DIR)

    # 基础配置
    base_config = {
        "min_ratio": 0.005,
        "feature_set": "advanced",
        "use_gpu": True,
        "max_samples": 50000,
    }

    # 1. 首先运行方法比较
    print("阶段1: 方法比较")
    methods = ["kmeans", "gmm", "dbscan"]
    runner.run_method_comparison(methods, base_config)

    # 2. 基于方法比较结果，对最佳方法进行参数优化
    print("\n阶段2: 参数优化")

    # 假设KMeans表现最好，优化聚类数目
    best_method = "kmeans"
    n_clusters_values = [80, 100, 120, 150, 180, 200]
    runner.run_parameter_sweep(
        best_method, "n_clusters", n_clusters_values, base_config
    )

    # 3. 使用最佳参数运行最终分析
    print("\n阶段3: 最终分析")
    best_n_clusters = 150  # 根据上一步结果调整

    final_config = {
        "name": "final_optimized_analysis",
        "method": best_method,
        "n_clusters": best_n_clusters,
        "min_ratio": 0.003,  # 更严格的过滤
        "feature_set": "all",  # 使用所有特征
        "use_gpu": False,
        "max_samples": 0,  # 不使用采样
    }

    final_result = runner.run_single_analysis(final_config)

    return final_result


if __name__ == "__main__":
    # 运行主函数
    main()

    # 如果要运行高级示例，取消注释下面的行
    # advanced_example()

    # 辅助方法：写入聚类结果到综合先验NPZ


def _safe_get(d, key, default=None):
    return d[key] if key in d else default


def _as_np_array(x):
    import numpy as np

    if x is None:
        return None
    try:
        return np.array(x)
    except Exception:
        return np.array([x], dtype=object)


def _result_to_npz_fields(result):
    fields = {}
    method = _safe_get(result, "method", "unknown")
    labels = _safe_get(result, "labels")
    n_clusters = _safe_get(result, "n_clusters")
    centers = _safe_get(result, "cluster_centers")
    sil = _safe_get(result, "silhouette_score")
    noise = _safe_get(result, "noise_ratio")
    elapsed = _safe_get(result, "elapsed_seconds")
    probs = _safe_get(result, "probabilities")
    best_linkage = _safe_get(result, "best_linkage")
    prefix = f"clustering_{method}"
    if labels is not None:
        fields[f"{prefix}_labels"] = _as_np_array(labels)
    if n_clusters is not None:
        fields[f"{prefix}_n_clusters"] = _as_np_array([n_clusters])
    if centers is not None:
        fields[f"{prefix}_centers"] = _as_np_array(centers)
    if sil is not None:
        fields[f"{prefix}_silhouette"] = _as_np_array([sil])
    if noise is not None:
        fields[f"{prefix}_noise_ratio"] = _as_np_array([noise])
    if elapsed is not None:
        fields[f"{prefix}_elapsed_seconds"] = _as_np_array([elapsed])
    if probs is not None:
        fields[f"{prefix}_probabilities"] = _as_np_array(probs)
    if best_linkage is not None:
        fields[f"{prefix}_best_linkage"] = _as_np_array([best_linkage])
    return fields


def _load_npz_as_dict(path):
    import numpy as np

    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _save_npz_from_dict(path, dct):
    import numpy as np

    np.savez(path, **dct)


def OpenLaneClusteringRunner__append_clustering_to_priors(self, priors_path, result):
    base = _load_npz_as_dict(str(priors_path))
    extra = _result_to_npz_fields(result)
    base.update(extra)
    _save_npz_from_dict(str(priors_path), base)


# 绑定到类
OpenLaneClusteringRunner._append_clustering_to_priors = (
    OpenLaneClusteringRunner__append_clustering_to_priors
)
