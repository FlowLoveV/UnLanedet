import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# =======================================

from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from omegaconf import DictConfig


def convert_dictconfig_to_dict(config):
    if isinstance(config, DictConfig):
        new_dict = {}
        for key, value in config.items():
            new_dict[key] = convert_dictconfig_to_dict(value)
        return new_dict
    else:
        return config


def CLRTransformsOpenLane(img_h, img_w):
    return [
        dict(
            name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0
        ),
        dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
        dict(
            name="Affine",
            parameters=dict(
                translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                rotate=(-10, 10),
                scale=(0.8, 1.2),
            ),
            p=0.7,
        ),
        dict(
            name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0
        ),
    ]


class GenerateLaneLineOpenLane(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.cfg = cfg
        self.training = training

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if transforms is None:
            transforms = CLRTransformsOpenLane(self.img_h, self.img_w)

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug["p"]
                if aug["name"] != "OneOf":
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=getattr(iaa, aug["name"])(**aug["parameters"]),
                        )
                    )
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf(
                                [
                                    getattr(iaa, aug_["name"])(**aug_["parameters"])
                                    for aug_ in aug["transforms"]
                                ]
                            ),
                        )
                    )
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        return [LineString(lane) for lane in lanes]

    def linestrings_to_lanes(self, line_strings):
        return [line.coords for line in line_strings]

    def sample_lane(self, points, sample_ys):
        points = np.array(points)
        if len(points) < 2:
            raise Exception("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]
        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3, len(points) - 1)
        )
        all_xs = interp(sample_ys)
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]
        return xs_outside_image, xs_inside_image

    def transform_annotation(self, anno, img_wh):
        """
        将原始标注转换为网络训练所需的 lane_line 格式

        ===================================================
        输入 anno["lanes"] 格式:
        - list of lists, 每个 lane 包含多个 (x, y) 坐标点
        - 坐标是图像像素坐标 (绝对坐标)
        - 例如: lanes = [[(100, 200), (150, 250), (200, 300)], ...]

        ===================================================
        输出 lanes 张量格式 (shape: [max_lanes, 6 + n_offsets]):
        - max_lanes: 最大车道线数量 (24)
        - n_offsets: 采样点数量 (72)

        各列含义:
        ---------------------------------------------------
        列索引 | 字段名称           | 形状说明                | 数值范围       | 单位/归一化
        -------|-------------------|------------------------|--------------|-----------
        0      | ignore_flag       | 是否忽略该车道线         | 0/1          | - (0=有效, 1=忽略)
        1      | is_lane           | 是否为车道线             | 0/1          | - (0=是车道线, 1=背景)
        2      | start_y           | 车道线起始 Y 坐标        | [0, 1]        | 归一化 (strip 索引 / n_strips)
        3      | start_x           | 车道线起始 X 坐标        | [0, 1]        | 归一化 (像素 / img_w)
        4      | theta             | 车道线平均角度           | [0, 1]        | 归一化 (角度/π)
        5      | length            | 车道线长度               | [0, 1]        | 归一化 (strip 数量 / n_strips)
        6+     | xs[0:n_offsets]  | 沿 Y 轴采样的 X 坐标序列  | [0, 1]        | 归一化 (像素 / img_w)
        ---------------------------------------------------

        关键说明:
        1. start_y, start_x, theta, xs 都是归一化值 [0, 1]
        2. length 是绝对值 (strip 数量)，范围 [1, 72]，未归一化！
        3. xs 序列从下到上按固定 Y 间隔采样，共 72 个点

        ===================================================
        输出 lane_endpoints 张量格式 (shape: [max_lanes, 2]):
        各列含义:
        ---------------------------------------------------
        列索引 | 字段名称           | 数值范围       | 单位/归一化
        -------|-------------------|--------------|-----------
        0      | endpoint_y        | [0, 1]        | 归一化 (strip 索引 / n_strips)
        1      | endpoint_x        | [0, 1]        | 归一化 (像素 / img_w)
        ---------------------------------------------------
        """
        old_lanes = anno["lanes"]
        # 过滤掉点数 <= 1 的车道线
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # 按 Y 坐标从下到上排序 (bottom to top)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]

        # 初始化 lanes 张量 (填充 -1e5 作为默认无效值)
        # shape: [max_lanes, 6 + n_offsets]
        lanes = (
            np.ones(
                (self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32
            )
            * -1e5
        )
        lanes_endpoints = np.ones((self.max_lanes, 2))
        # 默认: 忽略所有车道线 (ignore_flag=1)
        lanes[:, 0] = 1
        # 默认: 标记为背景 (is_lane=1, 0 表示车道线)
        lanes[:, 1] = 0

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            # 从稀疏标注点采样固定 Y 位置的 X 坐标
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys
                )
            except Exception:
                continue

            if len(xs_inside_image) <= 1:
                continue

            # 合并图像内外的所有采样点
            all_xs = np.hstack((xs_outside_image, xs_inside_image))

            # 标记该车道线有效 (ignore_flag=0, is_lane=1)
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1

            # 列 2: start_y - 车道线起始 Y 坐标 (strip 索引 / n_strips, 归一化到 [0,1])
            # 例如: 图像外有 5 个点, n_strips=71, 则 start_y = 5/71 ≈ 0.07
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips

            # 列 3: start_x - 车道线起始 X 坐标 (像素 / img_w, 归一化到 [0,1])
            # 例如: start_x = 100 / 800 = 0.125
            lanes[lane_idx, 3] = xs_inside_image[0] / self.img_w

            # 列 4: theta - 车道线平均角度 (归一化到 [0,1], 公式: 角度 / π)
            # 计算方法: 使用 atan(垂直距离 / 水平距离) / π
            # 注意: 这里的角度计算是从起始点到采样点的平均斜率
            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = (
                    math.atan(
                        i
                        * self.strip_size
                        / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                    )
                    / math.pi
                )
                # 将角度映射到 [0, 1] 范围 (处理负角度)
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas) if thetas else 0.0
            lanes[lane_idx, 4] = theta_far

            # 列 5: length - 车道线长度 (归一化到 [0,1], strip 数量 / n_strips)
            lanes[lane_idx, 5] = len(xs_inside_image) / self.n_strips

            # 列 6+: xs[0:n_offsets] - 采样点 X 坐标序列 (像素 / img_w, 归一化到 [0,1])
            # 共 72 个点, 从下到上按固定 Y 间隔采样
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs / self.img_w

            # lane_endpoints: 车道线终点坐标 (用于训练)
            # endpoint_y: (strip 索引 / n_strips, 归一化到 [0,1])
            # endpoint_x: (像素 / img_w, 归一化到 [0,1])
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1] / self.img_w

        new_anno = {"label": lanes, "lane_endpoints": lanes_endpoints}
        return new_anno

    def __call__(self, sample):
        img_org = sample["img"]
        img_h_curr, img_w_curr = img_org.shape[:2]
        is_preprocessed_img = img_h_curr == self.img_h
        global_cut_height = self.cfg.cut_height

        if not is_preprocessed_img and global_cut_height > 0:
            new_lanes = []
            for i in sample["lanes"]:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - global_cut_height))
                new_lanes.append(lanes)
            sample.update({"lanes": new_lanes})

        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        # Use a dummy mask because imgaug expects one if we pass segmentation_maps
        # but we handle segmentation manually later for speed
        dummy_mask_arr = np.zeros((img_h_curr, img_w_curr, 1), dtype=np.uint8)
        mask_org = SegmentationMapsOnImage(dummy_mask_arr, shape=img_org.shape)

        for i in range(30):
            try:
                if self.training:
                    # Pass dummy mask to keep imgaug happy if transforms require it
                    img, line_strings, _ = self.transform(
                        image=img_org.copy().astype(np.uint8),
                        line_strings=line_strings_org,
                        segmentation_maps=mask_org,
                    )
                else:
                    img, line_strings = self.transform(
                        image=img_org.copy().astype(np.uint8),
                        line_strings=line_strings_org,
                    )
                line_strings.clip_out_of_image_()
                new_anno = {"lanes": self.linestrings_to_lanes(line_strings)}
                annos = self.transform_annotation(
                    new_anno, img_wh=(self.img_w, self.img_h)
                )
                label = annos["label"]
                lane_endpoints = annos["lane_endpoints"]
                break
            except Exception:
                if (i + 1) == 30:
                    exit()

        # === Normalization (ImageNet) ===
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        sample["img"] = img
        sample["lane_line"] = label
        sample["lanes_endpoints"] = lane_endpoints
        sample["gt_points"] = new_anno["lanes"]

        # === Optimized Seg Mask (Downsample 8x, uint8) ===
        # Reduce memory usage significantly
        mask_scale = 8
        mask_h, mask_w = int(self.img_h // mask_scale), int(self.img_w // mask_scale)
        seg_map = np.zeros((mask_h, mask_w), dtype=np.uint8)

        lanes_points = new_anno["lanes"]
        for lane in lanes_points:
            if len(lane) < 2:
                continue
            pts = np.array([lane], dtype=np.float32) / mask_scale
            pts = pts.astype(np.int32)
            cv2.polylines(seg_map, pts, isClosed=False, color=1, thickness=1)

        sample["seg"] = seg_map.astype(np.int64)
        # ===============================================

        return sample
