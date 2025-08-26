# 此脚本是针对训练范围要求，如果标注范围与训练范围要求不同，则将过滤gt，生成对应mask后计算对应loss

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
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
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

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
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
                anchor, cls, qt, vis = self.layers[i](
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

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
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
        # box_now = self.decoder.decode_box(self.instance_bank.cached_anchor[0])
        # confidence = self.instance_bank.confidence
        # mask = confidence>0.2
        # confidence = confidence[mask]
        # box_now = box_now[mask[0]]
        # import mmcv
        # mmcv.dump(box_now,"./box2.pkl")
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output
    '''
    ['deformable', 'ffn', 'norm', 'refine', 
    'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 
    'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 
    'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 
    'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 
    'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
    '''  
    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        """
        计算损失函数，包括预测损失和去噪损失
        
        Args:
            model_outs: 模型输出，包含分类、预测、质量等信息
            data: 输入数据，包含真实标签
            feature_maps: 特征图（可选）
            
        Returns:
            output: 包含各层损失的字典
        """
        # ===================== 预测损失 ======================
        cls_scores = model_outs["classification"]  # 分类分数
        reg_preds = model_outs["prediction"]       # 回归预测
        quality = model_outs["quality"]            # 质量分数
        output = {}
        
        # 遍历每个解码器层
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            # 截取回归预测到指定维度
            reg = reg[..., : len(self.reg_weights)]
            
            # ==================== 采样器前的真实目标过滤 ====================
            # 在匹配分配前，先过滤真实目标：只保留标注范围内且标注了的类别
            filtered_gt_cls = []
            filtered_gt_reg = []
            
            for bs_id in range(len(data[self.gt_cls_key])):
                gt_cls = data[self.gt_cls_key][bs_id]
                gt_reg = data[self.gt_reg_key][bs_id]
                
                if len(gt_cls) == 0:
                    filtered_gt_cls.append(gt_cls)
                    filtered_gt_reg.append(gt_reg)
                    continue
                
                # 获取当前batch的过滤条件
                valid_mask = torch.ones(len(gt_cls), dtype=torch.bool, device=gt_cls.device)
                
                if (data['img_metas'][bs_id]).get('class_mask', None) is not None:
                    # 类别过滤：只保留标注的类别
                    class_mask = torch.tensor(data['img_metas'][bs_id]['class_mask']).to(gt_cls.device).bool()
                    cls_valid = class_mask[gt_cls]  # 检查每个真实类别是否被标注
                    valid_mask = torch.logical_and(valid_mask, cls_valid)
                    
                    # 范围过滤：只保留标注范围内的目标
                    range_mask = data['img_metas'][bs_id]['range_mask']  # [x_min, y_min, x_max, y_max]
                    gt_x, gt_y = gt_reg[:, 0], gt_reg[:, 1]  # 假设前两维是x,y坐标
                    range_valid = (range_mask[0] < gt_x) & (gt_x < range_mask[2]) & \
                                (range_mask[1] < gt_y) & (gt_y < range_mask[3])
                    valid_mask = torch.logical_and(valid_mask, range_valid)
                
                # 应用过滤
                filtered_gt_cls.append(gt_cls[valid_mask])
                filtered_gt_reg.append(gt_reg[valid_mask])
            
            # ==================== 采样器功能详解 ====================
            # 使用采样器进行预测和真实目标的匹配分配
            # 这是目标检测中的关键步骤：解决预测框与真实框的匹配问题
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,          # 预测分类分数 [batch_size, num_queries, num_classes]
                reg,          # 预测回归参数 [batch_size, num_queries, reg_dims]
                filtered_gt_cls,  # 过滤后的真实类别标签列表
                filtered_gt_reg,  # 过滤后的真实边界框列表
            )
            # 采样器功能详解：
            # 1. 计算匹配成本：分类成本 + 回归成本
            #    - 分类成本：基于Focal Loss计算预测类别与真实类别的成本
            #    - 回归成本：计算预测框与真实框之间的L1距离
            # 2. 匈牙利算法匹配：使用linear_sum_assignment找到最优分配
            #    - 每个预测查询(query)最多匹配一个真实目标
            #    - 每个真实目标最多匹配一个预测查询
            #    - 总成本最小化
            # 3. 输出结果：
            #    - cls_target: 分配给每个查询的目标类别 [batch_size, num_queries]
            #    - reg_target: 分配给每个查询的目标回归值 [batch_size, num_queries, reg_dims]
            #    - reg_weights: 每个回归参数的权重 [batch_size, num_queries, reg_dims]
            reg_target = reg_target[..., : len(self.reg_weights)]
            
            # 创建有效目标的掩码（非全零的回归目标）
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            # ==================== 预测结果的二次过滤（双重保障）====================
            # 注意：这是在采样器匹配后的额外过滤，确保预测结果也在有效范围内
            # 初始化类别掩码（默认全部为True）
            mask_cls = torch.ones_like(mask)
            
            # 如果存在类别掩码和距离掩码，对预测结果进行二次过滤
            if (data['img_metas'][0]).get('class_mask',None) is not None:
                # 遍历每个batch样本
                for bs_id in range(len(data['img_metas'])):
                    # === 第一步：类别过滤 ===
                    # 获取预定义的类别掩码（指定哪些类别需要保留）
                    class_mask = torch.tensor(data['img_metas'][bs_id]['class_mask']).to(cls.device).bool()
                    
                    # 获取距离范围掩码 [x_min, y_min, x_max, y_max]
                    range_mask =  data['img_metas'][bs_id]['range_mask']
                    
                    # 获取当前batch中每个检测框的预测类别索引
                    cls_pred = cls[bs_id].max(dim=-1)[1]  # 取最大值的索引作为预测类别
                    
                    # 根据预测类别索引，从class_mask中获取对应的掩码值
                    # 如果预测类别在允许的类别列表中，则cls_mask为True，否则为False
                    cls_mask = class_mask[cls_pred]

                    # 应用类别过滤：只保留允许的类别
                    mask_cls[bs_id] = torch.logical_and(mask_cls[bs_id], cls_mask)

                    # === 第二步：距离范围过滤 ===
                    # 获取当前batch的回归预测结果（包含位置信息）
                    reg_pred = reg[bs_id]
                    
                    # 构建距离范围掩码：检查预测位置是否在指定范围内
                    # range_mask格式: [x_min, y_min, x_max, y_max]
                    # 检查条件：x_min < pred_x < x_max AND y_min < pred_y < y_max
                    mask_range = (range_mask[0] < reg_pred[:,0]) & (reg_pred[:,0] < range_mask[2]) & \
                               (range_mask[1] < reg_pred[:,1]) & (reg_pred[:,1] < range_mask[3])

                    # 应用距离过滤：只保留在指定范围内的检测结果
                    mask_cls[bs_id] = torch.logical_and(mask_cls[bs_id], mask_range)
                    
            # 双重过滤策略总结：
            # 1. 采样器前过滤：过滤真实目标，影响匹配分配过程
            # 2. 采样器后过滤：过滤预测结果，确保损失计算的有效性
            # 两步过滤确保：只有标注范围内的标注类别参与训练

            # 计算正样本数量
            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            
            # 如果设置了分类阈值，进一步过滤掩码
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            # 展平掩码并应用过滤
            mask_cls = mask_cls.flatten(end_dim=1)
            cls = cls.flatten(end_dim=1)[mask_cls]
            cls_target = cls_target.flatten(end_dim=1)
            cls_target_ = cls_target[mask_cls]
            
            # 计算分类损失
            if mask_cls.sum()>0:
                cls_loss = self.loss_cls(cls, cls_target_, avg_factor=num_pos)
            else:
                cls_loss = (0*cls).sum()

            # 准备回归损失计算的数据
            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            
            # 处理NaN值
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            
            # 如果存在质量分数，也进行过滤
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            # 计算回归损失
            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            # 保存损失到输出字典
            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        # 如果没有去噪预测，直接返回
        if "dn_prediction" not in model_outs:
            return output

        # ===================== 去噪损失 ======================
        dn_cls_scores = model_outs["dn_classification"]  # 去噪分类分数
        dn_reg_preds = model_outs["dn_prediction"]       # 去噪回归预测

        # 准备去噪损失计算所需的数据
        (
            dn_valid_mask,    # 去噪有效掩码
            dn_cls_target,    # 去噪分类目标
            dn_reg_target,    # 去噪回归目标
            dn_pos_mask,      # 去噪正样本掩码
            reg_weights,      # 回归权重
            num_dn_pos,       # 去噪正样本数量
        ) = self.prepare_for_dn_loss(model_outs)
        
        # 遍历每个解码器层计算去噪损失
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            # 如果是临时去噪且达到单帧解码器层数，使用临时掩码
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

            # 计算去噪分类损失
            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            
            # 计算去噪回归损失
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            
            # 保存去噪损失到输出字典
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
            
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
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
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
