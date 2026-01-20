import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from .line_iou import line_iou, liou_loss
from .dynamic_assign import assign
from .roi_gather import ROIGather, LinearModule


class LLANetHead(nn.Module):
    def __init__(
        self,
        num_points=72,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        cfg=None,
        enable_category=True,
        enable_attribute=True,
        num_lane_categories=15,
        scale_factor=20.0,
        detailed_loss_logger=None,
    ):
        super(LLANetHead, self).__init__()
        self.cfg = cfg
        self.enable_category = enable_category
        self.enable_attribute = enable_attribute
        self.num_lane_categories = num_lane_categories
        self.scale_factor = scale_factor
        self.detailed_loss_logger = detailed_loss_logger

        self.img_w = 800 if cfg is None else self.cfg.img_w
        self.img_h = 320 if cfg is None else self.cfg.img_h
        self.num_lr_attributes = 4 if cfg is None else self.cfg.num_lr_attributes

        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        # Buffers
        self.register_buffer(
            "sample_x_indexs",
            (
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        self.register_buffer(
            "prior_feat_ys",
            torch.flip((1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]),
        )
        self.register_buffer(
            "prior_ys", torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        )

        self.prior_feat_channels = prior_feat_channels
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer("priors", init_priors)
        self.register_buffer("priors_on_featmap", priors_on_featmap)

        self.seg_conv = nn.Conv2d(
            self.prior_feat_channels * self.refine_layers, self.prior_feat_channels, 1
        )
        self.seg_decoder = PlainDecoder(cfg)

        reg_modules, cls_modules = [], []
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(
            self.prior_feat_channels,
            self.num_priors,
            self.sample_points,
            self.fc_hidden_dim,
            self.refine_layers,
            self.fc_hidden_dim,
        )
        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        # 设置分类层 bias，使得初始正样本概率极低（0.01），避免 Mode Collapse
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_layers.bias, bias_value)

        if self.enable_category:
            self.category_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            self.prototypes = nn.Parameter(
                torch.randn(self.num_lane_categories, self.fc_hidden_dim)
            )
            nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

        if self.enable_attribute:
            self.attribute_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            self.attribute_layers = nn.Linear(
                self.fc_hidden_dim, self.num_lr_attributes
            )

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(
            ignore_index=self.cfg.ignore_label, weight=weights
        )

        if self.enable_category:
            self.category_criterion = torch.nn.NLLLoss(reduction="sum")
        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction="sum")

        self.init_weights()

    def init_weights(self):
        # 分类层初始化：设置 bias 使得初始正样本概率极低
        for m in self.cls_layers.parameters():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                # bias已经在__init__中设置，这里如果不覆盖最好，或者再次确认
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias_value)
            else:
                nn.init.normal_(m, mean=0.0, std=1e-3)

        # 回归层初始化
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

        # Category 和 Attribute 层初始化
        if self.enable_category:
            for m in self.category_modules.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m, mean=0.0, std=1e-3)
        if self.enable_attribute:
            for m in self.attribute_layers.parameters():
                nn.init.normal_(m, mean=0.0, std=1e-3)

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        nn.init.normal_(self.prior_embeddings.weight, mean=0.0, std=0.01)

    def generate_priors_from_embeddings(self):
        device = self.prior_embeddings.weight.device
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)

        # [Fix]: Priors 初始化修正
        # Start Y: 设为 1.0 (图像底部)，而不是 0.1 (图像顶部)
        # 这样 ROIGather 才能在初始阶段提取到有效的车道线特征
        priors[:, 0] = 1.0
        priors[:, 1] = torch.linspace(
            0.0, 1.0, self.num_priors, device=device
        )  # Start X
        priors[:, 2] = 0.5  # Theta (0.5 * 180 = 90度，垂直)
        priors[:, 3] = 0.5  # Length
        priors[:, 4] = 0.0
        priors[:, 5] = 0.0

        # 生成先验形状 (直线)
        for i in range(self.num_priors):
            priors[i, 6:] = priors[i, 1]  # 简单的垂直线初始化

        priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.view(1, 1, -1, 1).expand(
            batch_size, num_priors, -1, 1
        )
        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid, align_corners=True)
        feature = feature.permute(0, 2, 1, 3).reshape(
            batch_size * num_priors, self.prior_feat_channels, self.sample_points, 1
        )
        return feature

    def forward(self, features, img_metas=None, **kwargs):
        batch_features = list(features[-self.refine_layers :])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]
        device = batch_features[-1].device

        if self.training:
            if self.prior_embeddings.weight.device != device:
                self.prior_embeddings = self.prior_embeddings.to(device)
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
        if self.priors.device != device:
            self.priors = self.priors.to(device)
            self.priors_on_featmap = self.priors_on_featmap.to(device)

        priors = self.priors.expand(batch_size, -1, -1)
        priors_on_featmap = self.priors_on_featmap.expand(batch_size, -1, -1)
        predictions_lists = []
        final_fc_features = None
        prior_features_stages = []
        prior_ys_expanded = (
            self.prior_ys.to(device)
            .view(1, 1, -1)
            .expand(batch_size, self.num_priors, -1)
        )

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs
            )
            prior_features_stages.append(batch_prior_features)
            fc_features = self.roi_gather(
                prior_features_stages, batch_features[stage], stage
            )
            fc_features = (
                fc_features.view(num_priors, batch_size, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            if stage == self.refine_layers - 1:
                final_fc_features = fc_features
            fc_features_flat = fc_features.view(
                batch_size * num_priors, self.fc_hidden_dim
            )

            cls_features = fc_features_flat.clone()
            reg_features = fc_features_flat.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)
            cls_logits = self.cls_layers(cls_features).reshape(batch_size, -1, 2)
            reg = self.reg_layers(reg_features).reshape(
                batch_size, -1, self.n_offsets + 1 + 2 + 1
            )

            predictions = priors.clone()
            predictions[..., :2] = cls_logits
            predictions[..., 2:5] += reg[..., :3]
            predictions[..., 5] = reg[..., 3]

            def unsqueeze_repeat(t):
                return t.unsqueeze(2).expand(-1, -1, self.n_offsets)

            pred_start_x = unsqueeze_repeat(predictions[..., 3])
            pred_theta = unsqueeze_repeat(predictions[..., 4])
            delta_y = 1.0 - prior_ys_expanded - unsqueeze_repeat(predictions[..., 2])
            cot_theta = 1.0 / torch.tan(pred_theta * math.pi + 1e-5)
            predictions[..., 6:] = (
                (pred_start_x * (self.img_w - 1)) + (delta_y * self.img_h * cot_theta)
            ) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            seg_out = self.seg_decoder(self.seg_conv(seg_features))
        else:
            seg_out = None

        category_out = None
        attribute_out = None
        if self.enable_category and final_fc_features is not None:
            cat_feat = final_fc_features.clone()
            for m in self.category_modules:
                cat_feat = m(cat_feat)
            category_out = self.scale_factor * torch.matmul(
                F.normalize(cat_feat, p=2, dim=-1),
                F.normalize(self.prototypes, p=2, dim=-1).transpose(0, 1),
            )

        if self.enable_attribute and final_fc_features is not None:
            attr_feat = final_fc_features.view(batch_size, self.num_priors, -1)
            for m in self.attribute_modules:
                attr_feat = m(attr_feat)
            attribute_out = self.attribute_layers(attr_feat)

        final_preds = predictions_lists[-1]
        if self.training:
            outputs = {"predictions_lists": predictions_lists}
            if seg_out is not None:
                outputs.update(**seg_out)
            if category_out is not None:
                outputs["category"] = category_out
            if attribute_out is not None:
                outputs["attribute"] = attribute_out
            return outputs
        else:
            ret_list = []
            batch_size = final_preds.shape[0]
            for i in range(batch_size):
                # ！！！注意：OpenLaneEvaluator 需要原始输出用于 NMS
                # 我们这里返回 predictions_to_pred 转换后的对象 (Lane Object)
                # 确保 Evaluator 能够处理 Lane Object 或者我们在这里不转
                # 按照你之前的代码逻辑，这里返回的是 Lane Object List
                sample_ret = {
                    "lane_lines": self.predictions_to_pred(final_preds[i].unsqueeze(0))
                }
                if category_out is not None:
                    sample_ret["category"] = category_out[i]
                if attribute_out is not None:
                    sample_ret["attribute"] = attribute_out[i]
                ret_list.append(sample_ret)
            return ret_list

    # 添加 predictions_to_pred 函数以支持 Evaluation
    def predictions_to_pred(self, predictions):
        # 简化版实现，只提取 line points
        # 实际使用时需要 Lane 类支持
        # 如果 Evaluator 是基于 Tensor 的，可以跳过这步
        # 这里为了兼容性保留你原来 head 里的逻辑
        lanes = []
        for lane in predictions:
            lane = lane.detach().cpu().numpy()
            lanes.append(lane)  # 直接返回 numpy 数组，交给 Evaluator 处理
        return lanes

    def loss(self, outputs, batch):
        """Vectorized Loss Calculation Optimized for GPU"""
        predictions_lists = outputs["predictions_lists"]
        targets_list = batch["lane_line"]  # List of [N, Dim] tensors
        batch_lane_categories = batch.get("lane_categories", None)
        batch_lane_attributes = batch.get("lane_attributes", None)

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        device = self.priors.device
        batch_size = len(targets_list)

        # ============================================================
        # 1. 预处理 Targets: Pad to Tensor (B, Max_Lanes, Dim)
        # ============================================================
        max_lanes = 0
        target_dim = 0
        if len(targets_list) > 0:
            lengths = [len(t) for t in targets_list]
            max_lanes = max(lengths) if lengths else 0
            if max_lanes > 0:
                target_dim = targets_list[0].shape[1]

        max_lanes = max(max_lanes, 1)
        if target_dim == 0:
            target_dim = 6 + 72

        batch_targets = torch.zeros((batch_size, max_lanes, target_dim), device=device)
        batch_masks = torch.zeros((batch_size, max_lanes), device=device)

        for i, t in enumerate(targets_list):
            num_t = t.shape[0]
            if num_t > 0:
                valid_mask = t[:, 1] == 1
                valid_t = t[valid_mask]
                num_valid = valid_t.shape[0]
                if num_valid > 0:
                    num_valid = min(num_valid, max_lanes)
                    batch_targets[i, :num_valid] = valid_t[:num_valid]
                    batch_masks[i, :num_valid] = 1

        # ============================================================
        # 2. 向量化训练循环
        # ============================================================
        total_cls_loss = 0.0
        total_reg_xytl_loss = 0.0
        total_iou_loss = 0.0
        total_category_loss = 0.0
        total_attribute_loss = 0.0

        # 用于记录分项损失的累加器
        total_loss_y_sum = 0.0
        total_loss_x_sum = 0.0
        total_loss_theta_sum = 0.0
        total_loss_len_sum = 0.0
        total_stage_count = 0

        for stage in range(self.refine_layers):
            predictions = predictions_lists[stage]

            with torch.no_grad():
                assigned_mask, assigned_ids = assign(
                    predictions,
                    batch_targets,
                    batch_masks,
                    self.img_w,
                    self.img_h,
                    self.cfg,
                )

            # --- Classification Loss ---
            cls_targets = assigned_mask.long().view(-1)
            pred_cls_flat = predictions[..., :2].view(-1, 2)
            num_positives = assigned_mask.sum()
            cls_norm = max(num_positives.item(), 1.0)

            stage_cls_loss = cls_criterion(pred_cls_flat, cls_targets).sum()
            total_cls_loss += stage_cls_loss / cls_norm

            if num_positives > 0:
                # --- 选取正样本 ---
                pos_preds = predictions[assigned_mask]
                batch_idx, prior_idx = torch.where(assigned_mask)
                target_idx = assigned_ids[batch_idx, prior_idx]
                pos_targets = batch_targets[batch_idx, target_idx]

                # --- Regression Loss (Absolute Coordinates) ---
                reg_pred = pos_preds[:, 2:6]
                reg_target = pos_targets[:, 2:6].clone()

                with torch.no_grad():
                    pred_starts = torch.clamp(
                        (reg_pred[:, 0] * self.n_strips).round().long(),
                        0,
                        self.n_strips,
                    )
                    target_starts = (reg_target[:, 0] * self.n_strips).round().long()
                    start_diff = pred_starts - target_starts
                    if reg_target[:, 3].max() > 1.0:
                        reg_target[:, 3] -= start_diff.float()
                    else:
                        reg_target[:, 3] -= start_diff / self.n_strips

                # Start Y: 0~72
                pred_y_abs = reg_pred[:, 0] * self.n_strips
                target_y_abs = reg_target[:, 0] * self.n_strips
                # Start X: 0~800
                pred_x_abs = reg_pred[:, 1] * (self.img_w - 1)
                target_x_abs = reg_target[:, 1] * (self.img_w - 1)
                # Theta: 0~180 (角度制)
                # Target 的 theta 已经是归一化的 0-1（atan/pi），需要乘 180
                pred_theta_abs = reg_pred[:, 2] * 180
                target_theta_abs = reg_target[:, 2] * 180
                # Length: 0~72
                pred_len_abs = reg_pred[:, 3] * self.n_strips
                target_len_abs = reg_target[:, 3]
                # 计算 Smooth L1
                loss_y = F.smooth_l1_loss(pred_y_abs, target_y_abs, reduction="none")
                loss_x = F.smooth_l1_loss(pred_x_abs, target_x_abs, reduction="none")
                loss_theta = F.smooth_l1_loss(
                    pred_theta_abs, target_theta_abs, reduction="none"
                )
                loss_len = F.smooth_l1_loss(
                    pred_len_abs, target_len_abs, reduction="none"
                )

                # 【权重平衡】
                # 降低 X 的权重 (0.5)，防止数值过大主导梯度
                # 其他保持 1.0
                reg_weights = torch.tensor([1.0, 0.5, 1.0, 1.0], device=device)

                loss_components = torch.stack(
                    [loss_y, loss_x, loss_theta, loss_len], dim=1
                )

                # Sum and Normalize by num_positives
                total_reg_xytl_loss += (loss_components * reg_weights).sum() / max(
                    num_positives, 1
                )

                # 累加分项损失用于记录（如果提供了 logger）
                if self.detailed_loss_logger is not None:
                    total_loss_y_sum += loss_y.mean().item()
                    total_loss_x_sum += loss_x.mean().item()
                    total_loss_theta_sum += loss_theta.mean().item()
                    total_loss_len_sum += loss_len.mean().item()
                    total_stage_count += 1

                # --- IoU Loss ---
                line_pred = pos_preds[:, 6:] * (self.img_w - 1)
                line_target = pos_targets[:, 6:] * (self.img_w - 1)

                total_iou_loss += liou_loss(
                    line_pred, line_target, self.img_w, length=15
                )

                # --- Category & Attribute Loss ---
                if stage == self.refine_layers - 1:
                    if self.enable_category and "category" in outputs:
                        cat_preds = outputs["category"][assigned_mask]
                        if isinstance(batch_lane_categories, torch.Tensor):
                            cat_targets = batch_lane_categories[batch_idx, target_idx]
                            total_category_loss += self.category_criterion(
                                cat_preds.log_softmax(dim=-1), cat_targets
                            ) / max(batch_size, 1)

                    if self.enable_attribute and "attribute" in outputs:
                        attr_preds = outputs["attribute"][assigned_mask]
                        if isinstance(batch_lane_attributes, torch.Tensor):
                            attr_targets = batch_lane_attributes[batch_idx, target_idx]
                            total_attribute_loss += self.attribute_criterion(
                                attr_preds.log_softmax(dim=-1), attr_targets
                            ) / max(batch_size, 1)

        # Seg Loss 计算（移至记录之前）
        seg_loss = torch.tensor(0.0, device=device)
        if outputs.get("seg", None) is not None and batch.get("seg", None) is not None:
            seg_pred = outputs["seg"]
            if seg_pred.shape[-2:] != batch["seg"].shape[-2:]:
                seg_pred = F.interpolate(
                    seg_pred,
                    size=batch["seg"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            seg_loss = self.criterion(
                F.log_softmax(seg_pred, dim=1), batch["seg"].long()
            )

        # 记录分项损失（如果提供了 logger）
        if self.detailed_loss_logger is not None and total_stage_count > 0:
            detailed_loss_dict = {
                "loss_start_y": total_loss_y_sum / total_stage_count,
                "loss_start_x": total_loss_x_sum / total_stage_count,
                "loss_theta": total_loss_theta_sum / total_stage_count,
                "loss_length": total_loss_len_sum / total_stage_count,
                "reg_xytl_loss": total_reg_xytl_loss.item() / self.refine_layers,
                "cls_loss": total_cls_loss.item() / self.refine_layers,
                "iou_loss": total_iou_loss.item() / self.refine_layers,
                "loss_category": total_category_loss.item(),
                "loss_attribute": total_attribute_loss.item(),
                "seg_loss": seg_loss.item(),
            }
            self.detailed_loss_logger.log(None, detailed_loss_dict)
        # 3. 汇总 Loss
        losses = {}
        losses["cls_loss"] = (
            total_cls_loss / self.refine_layers * self.cfg.cls_loss_weight
        )
        losses["reg_xytl_loss"] = (
            total_reg_xytl_loss / self.refine_layers * self.cfg.xyt_loss_weight
        )
        losses["iou_loss"] = (
            total_iou_loss / self.refine_layers * self.cfg.iou_loss_weight
        )
        losses["loss_category"] = total_category_loss * self.cfg.category_loss_weight
        losses["loss_attribute"] = total_attribute_loss * self.cfg.attribute_loss_weight

        losses["seg_loss"] = seg_loss * self.cfg.seg_loss_weight

        return losses
