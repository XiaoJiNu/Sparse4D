# Sparse4D 模型整体架构分析

> 参考论文：《Sparse4D: Multi-view 3D Object Detection with Sparse Spatio-Temporal Queries》，结合仓库中的 `projects/mmdet3d_plugin` 实现。

## 1. 总览：从多视角到稀疏时空查询

Sparse4D 以多视角相机图像作为输入，通过 CNN + FPN 提取多尺度特征，再借助稀疏的 3D 查询（query）完成跨视角、跨时间的特征聚合与预测。整体流程如下：

1. **图像编码**：`ResNet-50` + `FPN` 输出 4 个尺度的特征图，每个尺度通道数为 `embed_dims=256`，形状 `[B, N_cam, 256, H_l, W_l]`。
2. **深度辅助分支（可选）**：`DenseDepthNet` 对前 `num_depth_layers=3` 个尺度预测单通道深度图，监督来自 `gt_depth`。
3. **实例库（InstanceBank）初始化查询**：从 `nuscenes_kmeans900.npy` 中取 `num_anchor=900` 个稀疏锚点（11 维状态向量），并维护 `num_temp_instances=600` 个历史实例，实现跨帧记忆。
4. **Transformer 风格的稀疏解码器**：每轮包含以下子模块：
   - `temp_gnn`：与历史实例做时序注意力；
   - `gnn`：当前实例间自注意力；
   - `deformable`：`DeformableFeatureAggregation` 进行跨视角特征采样；
   - `ffn` + `norm`：前馈网络、自适应归一化；
   - `refine`：`SparseBox3DRefinementModule` 更新锚点并输出分类分数与质量估计。
5. **输出解码**：`SparseBox3DDecoder` 将 `[x, y, z, log w, log l, log h, sin θ, cos θ, vx, vy, vz]` 解码为真实尺度/角度的 3D 框并筛选 Top-K 结果。

最终输出的检测结果结构为：

- `boxes_3d`: `[N_det, 10]`，包含中心、长宽高、航向角、速度；
- `scores_3d`: `[N_det]`，类别得分；
- `labels_3d`: `[N_det]`，类别索引；
- `instance_ids`（推理时可选）：跨帧追踪 ID。

## 2. 输入与特征维度梳理

| 模块 | 输入形状 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| `extract_feat` | `[B, N_cam, 3, H, W]`，默认 `H=256, W=704, N_cam=6` | `[L, B, N_cam, 256, H_l, W_l]` | L=4 个 FPN 尺度，空间尺寸依次约为 `(64×176, 32×88, 16×44, 8×22)` |
| `DepthBranch` | 同上 | `[3, B*N_cam, 1, H_l, W_l]` | 预测多尺度深度，用于辅助监督 |
| `InstanceBank.get` | - | `instance_feature: [B, 900, 256]`<br>`anchor: [B, 900, 11]`<br>`temp_*: [B, 600, •]` | 历史缓存提供额外查询供 `temp_gnn` 使用 |
| `anchor_encoder` | `anchor: [B, N, 11]` | `[B, N, 256]` | 将 3D 框参数编码成嵌入，用作位置编码 |
| `temp_gnn`/`gnn` | `query/value: [B, N, 256]` | `[B, N, 256]` | `num_heads = num_groups = 8`，若 `decouple_attn=True` 则拼接 `anchor_embed` 形成 512 维内部表示 |
| `deformable` | `instance_feature: [B, N, 256]`<br>`anchor: [B, N, 11]`<br>`feature_maps` | 输出 `residual` 形状 `[B, N, 256]`，与原特征拼接得到 `[B, N, 512]` | 通过关键点采样（`num_pts=7`）融合多视角特征 |
| `ffn` | `[B, N, 512]` | `[B, N, 256]` | `AsymmetricFFN`：两层线性 + ReLU，支持残差连接 |
| `refine` | `instance_feature: [B, N, 256]` + `anchor_embed` | `anchor': [B, N, 11]`<br>`cls: [B, N, num_cls]`<br>`quality: [B, N, 2]` | 回归残差直接加到 anchor state，尺寸指数化，yaw 归一化，速度按时间间隔平移 |
| `decoder.decode` | `classification/prediction` 列表 | 每层输出 `[B, N, num_cls]` 与 `[B, N, 11]` | 取最后一层做 Top-K（默认 300）并恢复物理量 |

> 注：anchor 的 11 维顺序由 `projects/mmdet3d_plugin/core/box3d.py` 定义，依次为 `(x, y, z, log w, log l, log h, sin θ, cos θ, vx, vy, vz)`。

## 3. 关键子模块深挖

### 3.1 图像骨干与深度分支

- `img_backbone`: 标准的 `ResNet-50`，输出索引 `(0,1,2,3)` 层；启用 `with_cp=True` 以节省显存。
- `img_neck`: `FPN` 聚合为统一的 256 维特征。
- `depth_branch`: `DenseDepthNet` 仅在训练中计算 `loss_dense_depth`（指数输出保证正值），焦距由 meta 中的 `focal` 标准化后恢复。

### 3.2 InstanceBank：稀疏查询与时序缓存

- **初始化**：900 个锚点来自离线 k-means；`instance_feature` 是可学习参数，形状 `[900, 256]`。
- **时序缓存**：
  - `cache` 会根据上一帧得分选择 `num_temp_instances=600` 个最可靠的实例，并保存其 anchor / feature；
  - 通过 `anchor_handler`（`SparseBox3DKeyPointsGenerator`）利用位姿矩阵将历史 anchor 投影至当前坐标系，实现时间对齐；
  - 在 `get_instance_id` 中结合置信度生成连续的实例 ID，用于跟踪输出。
- **更新策略**：`update` 阶段将新预测与历史缓存按得分融合：先选取 top-N（`num_anchor - num_temp_instances`）作为基础，再与缓存合并，维持稀疏性。

### 3.3 注意力序列与操作顺序

配置中 `num_decoder=6`、`num_single_frame_decoder=1`，`operation_order` 经过裁剪后为：

```
[deformable, ffn, norm, refine,
 temp_gnn, gnn, norm, deformable, ffn, norm, refine] × 5 次
```

即首个循环省略 `gnn`（避免重复与初始化冲突），后续 5 个循环均包含时序 + 当前帧注意力。

#### a. `temp_gnn`

- 输入：`query = [B, N, 256]`，`key=value = temp_instance_feature = [B, 600, 256]`。
- 作用：跨帧信息注入；若上一帧缓存为空则跳过。
- 注意力掩码：仅当没有缓存时使用由去噪模块生成的 `attn_mask`。

#### b. `gnn`

- 输入：当前实例自身；如果开启 `decouple_attn`，则将 anchor 编码与实例特征拼接（512 维）后再做 `MultiheadAttention`。
- 输出：同维度 `[B, N, 256]`，保持稀疏查询间的信息交互。

#### c. `deformable`

- 关键点生成：
  - 固定 7 个关键点（中心 + 六个轴向偏移），并可额外学习 `num_learnable_pts=6`（配置中启用）。
  - 使用 anchor 尺寸经过指数和旋转矩阵映射至三维空间，支持速度补偿。
- 特征采样：
  - 将关键点投影至 6 个相机，利用 `projection_mat`（`[B, N_cam, 4, 4]`）得到归一化坐标；
  - 若 `use_deformable_func=True`，调用自定义 CUDA 内核 `DeformableAggregationFunction`，效率更高；否则回退到 `grid_sample`。
- 权重生成：`weights_fc` 将 `[B, N, 256]` 投影为 `num_groups × num_levels × num_pts` 权重，经过 Softmax + dropout，实现跨尺度融合。
- 输出：与 `instance_feature` 残差拼接（`residual_mode='cat'`），因此给 `ffn` 的输入通道为 512。

#### d. `ffn` 与 `norm`

- `AsymmetricFFN`：两层线性（512→1024→256），中间 ReLU + Dropout，具备可配置的预归一化；
- `norm_layer`: `LayerNorm`，维度 256。

#### e. `refine`

- 对 anchor state 进行残差更新，并输出：
  - `prediction`: `[B, N, 11]`，包含更新后的 state；
  - `classification`: `[B, N, num_cls]`（配置中 `num_cls=10`）；
  - `quality`: `[B, N, 2]`（centerness + yawness），仅在 `with_quality_estimation=True` 时返回。
- 若 `time_interval` 存在，速度部分根据时间差折算成位移后再恢复为速度，使得锚点在时间维度上连续。
- 单帧解码器结束后，会调用 `InstanceBank.update` 刷新缓存，为后续时序块提供输入。

### 3.4 去噪训练（Denoising）

- `SparseBox3DTarget.get_dn_anchors` 会为每个 GT 复制 `num_dn_groups=5` 组噪声锚点：
  - 正样本噪声（坐标扰动 `dn_noise_scale`）；
  - 可选负样本（大噪声），增强鲁棒性；
  - 输出 `dn_attn_mask`，确保不同去噪组之间注意力隔离。
- 模型前向时将去噪 anchor 拼接到查询后部，引入额外的 `dn_prediction`、`dn_classification`。
- `loss` 中调用 `prepare_for_dn_loss` 过滤有效样本并计算额外的去噪损失，平均因子为有效去噪正样本数量。

### 3.5 解码与后处理

- `SparseBox3DDecoder.decode`：
  - 对指定层（默认最后一层）的分类分数做 Sigmoid；
  - Top-K 选择（默认 `num_output=300`）；
  - 若启用质量估计则乘以 `centerness`；
  - `decode_box` 将 log 尺寸指数化，并用 `atan2(sin, cos)` 恢复航向角；
  - 若 `instance_id` 存在，附带返回，支持在线跟踪。

## 4. 训练与评估流程补充

- **数据管线**：`NuScenesSparse4DAdaptor` 会整理多视角图像、投影矩阵、时间戳、实例 ID 等信息，供 `InstanceBank` 与 `DeformableFeatureAggregation` 使用。
- **优化器**：`AdamW(lr=6e-4)`，图像骨干使用 `lr_mult=0.5`，梯度裁剪 `max_norm=25`。
- **学习率策略**：`CosineAnnealing` + 500 次线性 warmup，最小比率 `1e-3`。
- **混合精度**：配置中设定 `loss_scale=32` 的静态缩放。
- **评估**：在 `evaluation.pipeline` 中仅保留图像与投影矩阵，可选保存可视化结果。

## 5. 与论文要点的对应关系

| 论文中的核心概念 | 代码实现要点 |
| --- | --- |
| 稀疏时空查询（Sparse Spatio-Temporal Queries） | `InstanceBank` + `temp_gnn` 对历史实例进行缓存与跨帧注意力；查询数量固定为 900，但仅保留高分实例进入下一帧。 |
| 多视角稀疏采样（Sparse Multi-view Sampling） | `DeformableFeatureAggregation` 根据 3D 框关键点预测对应的 2D 像素，用自定义 CUDA 或 `grid_sample` 做稀疏采样，并基于可学习权重融合多尺度、多视角特征。 |
| 动态更新的对象状态 | `SparseBox3DRefinementModule` 输出 11 维状态，直接回写 anchor，配合 `InstanceBank.update` 形成在线的状态跟踪。 |
| 去噪训练（DN）增强鲁棒性 | `SparseBox3DTarget` 中的 `get_dn_anchors`、`prepare_for_dn_loss` 与 head 中的拼接/分离逻辑完整复现论文提出的去噪策略。 |
| 质量估计辅助（centerness/yawness） | `with_quality_estimation=True` 时在 `refine` 中增加质量分支，解码阶段乘以 centerness 以提高排序准确性。 |

## 6. 小结

Sparse4D 通过“稀疏查询 + 可学习实例库 + 多视角可变形聚合 + 时序注意力”的组合，实现了高效的多摄像头 3D 检测与跟踪。一方面依赖稀疏查询减少计算开销（相较于逐像素 dense 方法），另一方面利用 `InstanceBank` 维护对象记忆，使得跨帧信息自然注入模型。配合去噪训练和质量估计，模型在 nuScenes 上取得较高精度（参见配置文件头部记录的 NDS/mAP）。

继续深入时，可从以下两个方向延展：

1. **算子级优化**：当启用自定义 CUDA (`use_deformable_func=True`) 时，需要使用 `python3 setup.py develop` 重新编译；评估差异可参考 `projects/mmdet3d_plugin/ops/deformable_aggregation`。
2. **参数敏感性**：如需调整查询数量或去噪组数，需同步修改 `InstanceBank`、`SparseBox3DTarget` 等的配置，保持维度一致。
