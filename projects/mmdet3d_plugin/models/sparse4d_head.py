# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],  # 输入特征图，维度为 [B, C, H, W] 或特征图列表
        metas: dict,  # 包含图像元数据和GT标注信息的字典
    ):
        """
        Sparse4D头部的前向传播函数
        
        输入:
            feature_maps: 特征图张量或列表，维度为 [B, C, H, W]
            metas: 元数据字典，包含:
                - img_metas: 图像元信息
                - gt_labels_3d: 3D标签真值，维度为 [B, N_gt]
                - gt_bboxes_3d: 3D边框真值，维度为 [B, N_gt, bbox_dim]
                
        输出:
            output: 包含预测结果的字典，包括:
                - classification: 分类预测结果列表，每个元素维度为 [B, N_anchor, num_classes] 
                - prediction: 回归预测结果列表，每个元素维度为 [B, N_anchor, bbox_dim]
                - quality: 质量预测结果列表（可选）
                - dn_prediction: 去噪预测结果列表（训练时且启用去噪时）
                - dn_classification: 去噪分类结果列表（训练时且启用去噪时）
                - dn_reg_target: 去噪回归目标（训练时且启用去噪时）
                - dn_cls_target: 去噪分类目标（训练时且启用去噪时）
                - instance_id: 实例ID（推理时）
                
        实现逻辑:
        1. 从实例库获取实例特征和锚点
        2. 如果是训练模式，准备去噪训练数据
        3. 通过多层transformer进行特征处理
        4. 分离学习实例和噪声实例的预测结果
        5. 缓存当前实例用于时序建模
        """
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # ========= 准备去噪训练 ============
        # 1. 获取去噪元数据：噪声锚点和对应的GT
        # 2. 拼接可学习实例和噪声实例
        # 3. 获取注意力掩码
        attn_mask = None  # 注意力掩码，维度为 [N_total, N_total]，控制实例间的注意力交互
        dn_metas = None   # 去噪元数据，包含噪声锚点和目标
        temp_dn_reg_target = None  # 时序去噪回归目标
        # 只在训练模式且采样器支持去噪时生成去噪数据
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            # 检查是否有实例ID信息，用于时序追踪
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            # 调用采样器生成去噪数据：噪声锚点、目标、掩码等
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],    # GT分类标签，维度 [B, N_gt]
                metas[self.gt_reg_key],    # GT回归标签，维度 [B, N_gt, bbox_dim]
                gt_instance_id,            # GT实例ID（可选）
            )
        if dn_metas is not None:
            (
                dn_anchor,      # 去噪锚点，维度为 [B, N_dn, anchor_dim]
                dn_reg_target,  # 去噪回归目标，维度为 [B, N_dn, bbox_dim]
                dn_cls_target,  # 去噪分类目标，维度为 [B, N_dn]
                dn_attn_mask,   # 去噪注意力掩码，维度为 [N_dn, N_dn]
                valid_mask,     # 有效掩码，维度为 [B, N_dn]
                dn_id_target,   # 去噪ID目标，维度为 [B, N_dn]（可选）
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]  # 去噪锚点数量
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # 分离学习实例和噪声实例的预测结果
        # 这里是model_outs中dn_prediction生成的关键位置
        if dn_metas is not None:
            # 提取去噪分类预测结果：取后num_dn_anchor个实例的分类预测
            # 维度：每个元素为 [B, N_dn, num_classes]
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            # 保留前num_free_instance个学习实例的分类预测
            # 维度：每个元素为 [B, N_free, num_classes]
            classification = [x[:, :num_free_instance] for x in classification]
            # 提取去噪回归预测结果：取后num_dn_anchor个实例的回归预测
            # 维度：每个元素为 [B, N_dn, bbox_dim] - 这就是dn_prediction！
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            # 保留前num_free_instance个学习实例的回归预测
            # 维度：每个元素为 [B, N_free, bbox_dim]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            # 将去噪相关的预测结果添加到输出字典中
            # 这里是dn_prediction被添加到model_outs的地方
            output.update(
                {
                    "dn_prediction": dn_prediction,        # 去噪回归预测，列表，每个元素维度 [B, N_dn, bbox_dim]
                    "dn_classification": dn_classification,  # 去噪分类预测，列表，每个元素维度 [B, N_dn, num_classes]
                    "dn_reg_target": dn_reg_target,        # 去噪回归目标，维度 [B, N_dn, bbox_dim]
                    "dn_cls_target": dn_cls_target,        # 去噪分类目标，维度 [B, N_dn]
                    "dn_valid_mask": valid_mask,           # 去噪有效掩码，维度 [B, N_dn]
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        """
        计算损失函数
        
        输入:
            model_outs: 模型输出字典，包含:
                - classification: 分类预测列表，每个元素维度 [B, N_anchor, num_classes]
                - prediction: 回归预测列表，每个元素维度 [B, N_anchor, bbox_dim]
                - quality: 质量预测列表（可选）
                - dn_prediction: 去噪回归预测列表（仅在启用去噪时存在）
                - dn_classification: 去噪分类预测列表（仅在启用去噪时存在）
                - 其他去噪相关字段
            data: 包含GT数据的字典
            feature_maps: 特征图（可选）
            
        输出:
            output: 损失字典，包含各decoder层的分类和回归损失
            
        实现逻辑:
        1. 计算常规预测的损失（所有decoder层）
        2. 如果存在dn_prediction，计算去噪损失
        3. 返回所有损失的字典
        """
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        # 检查是否存在去噪预测结果
        # 条件成立的情况：
        # 1. 推理模式（self.training=False）时，不会生成dn_prediction
        # 2. 训练模式但采样器不支持去噪（没有get_dn_anchors方法）
        # 3. 训练模式但当前批次没有GT数据，导致dn_metas为None
        # 4. 训练模式但由于其他原因导致去噪数据生成失败
        if "dn_prediction" not in model_outs:
            return output  # 直接返回常规预测的损失，不计算去噪损失

        # ===================== 去噪损失计算 ======================
        # 获取去噪预测结果
        dn_cls_scores = model_outs["dn_classification"]  # 去噪分类预测，列表，每个元素维度 [B, N_dn, num_classes]
        dn_reg_preds = model_outs["dn_prediction"]       # 去噪回归预测，列表，每个元素维度 [B, N_dn, bbox_dim]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """
        为去噪损失计算准备数据
        
        输入:
            model_outs: 模型输出字典，包含去噪相关数据
            prefix: 前缀字符串，用于支持时序去噪（如"temp_"）
            
        输出:
            返回元组，包含:
            - dn_valid_mask: 去噪有效掩码，维度 [B*N_dn] -> bool类型
            - dn_cls_target: 有效的去噪分类目标，维度 [N_valid]
            - dn_reg_target: 正样本的去噪回归目标，维度 [N_pos, bbox_dim]
            - dn_pos_mask: 正样本掩码，维度 [N_valid] -> bool类型
            - reg_weights: 回归损失权重，维度 [N_pos, bbox_dim]
            - num_dn_pos: 有效正样本数量，标量
            
        实现逻辑:
        1. 获取去噪数据并展平为1D
        2. 过滤出有效的数据（valid_mask=True）
        3. 过滤出正样本（cls_target>=0）
        4. 准备回归损失的权重
        5. 计算有效正样本数量用于平均
        """
        # 1. 获取去噪数据并展平为1D形状
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)  # [B, N_dn] -> [B*N_dn]
        
        # 2. 使用valid_mask过滤出有效的去噪数据
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1  # [B, N_dn] -> [B*N_dn]
        )[dn_valid_mask]  # 只保留有效的目标 -> [N_valid]
        
        # 3. 获取有效的回归目标，并截取到指定维度
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1  # [B, N_dn, bbox_dim] -> [B*N_dn, bbox_dim]
        )[dn_valid_mask][..., : len(self.reg_weights)]  # [N_valid, bbox_dim] 截取到指定维度
        
        # 4. 找出正样本（类别标签>=0的样本）
        dn_pos_mask = dn_cls_target >= 0  # [N_valid] -> bool掩码，标记正样本
        
        # 5. 只保留正样本的回归目标
        dn_reg_target = dn_reg_target[dn_pos_mask]  # [N_pos, bbox_dim]
        
        # 6. 准备回归损失的权重，为每个正样本复制一份权重
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1  # [N_pos, bbox_dim]
        )
        
        # 7. 计算有效正样本数量，用于损失平均，最小为1.0避免除零
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        """
        后处理函数，将原始预测结果解码为最终的检测结果
        
        输入:
            model_outs: 模型输出字典，包含:
                - classification: 分类预测列表
                - prediction: 回归预测列表 
                - instance_id: 实例ID（可选）
                - quality: 质量预测（可选）
            output_idx: 使用哪个解码器层的输出，-1表示最后一层
            
        输出:
            解码后的检测结果，包含边框、分数、标签等
            
        注意:
            - 这个函数不处理dn_prediction，因为去噪预测只用于训练
            - 只在推理时调用，训练时不会调用这个函数
        """
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
