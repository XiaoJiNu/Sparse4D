"""
mAP: 0.4647
mATE: 0.5403
mASE: 0.2623
mAOE: 0.4590
mAVE: 0.2198
mAAE: 0.2059
NDS: 0.5636
Eval time: 176.9s

Per-class results:
Object Class    AP  ATE ASE AOE AVE AAE
car 0.668   0.357   0.142   0.054   0.184   0.195
truck   0.394   0.528   0.187   0.052   0.163   0.210
bus 0.451   0.681   0.196   0.070   0.383   0.243
trailer 0.185   0.971   0.247   0.634   0.175   0.202
construction_vehicle    0.122   0.879   0.496   1.200   0.136   0.406
pedestrian  0.559   0.517   0.287   0.513   0.282   0.151
motorcycle  0.497   0.462   0.238   0.536   0.293   0.236
bicycle 0.426   0.441   0.257   0.951   0.142   0.004
traffic_cone    0.697   0.275   0.299   nan nan nan
barrier 0.648   0.292   0.275   0.122   nan nan
"""

"""
Per-class results:
            AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT  ML  FAF     TP      FP  FN  IDS FRAG TID    LGD
bicycle     0.444   1.169   0.533   0.733   1993    0.389   0.566   53  57  19.3    1059    283 931 3   8   1.60    1.75
bus         0.559   1.175   0.626   0.824   2112    0.515   0.751   42  35  14.8    1321    233 790 1   20  1.13    1.95
car         0.678   0.755   0.733   0.819   58317   0.599   0.470   2053    1073    134.2   42626   7706    15565   126 295 0.76    1.03
motorcy     0.522   1.060   0.609   0.823   1977    0.497   0.564   50  38  15.7    1194    211 773 10  17  1.97    2.17
pedestr     0.548   1.059   0.652   0.791   25423   0.506   0.678   677 467 77.6    16274   3404    8854    295 225 1.33    1.85
trailer     0.136   1.603   0.383   0.403   2425    0.154   0.981   30  79  52.6    926 553 1496    3   13  1.49    2.64
truck       0.454   1.132   0.577   0.691   9650    0.399   0.594   210 214 45.7    5569    1723    4078    3   50  1.35    1.85

Aggregated results:
AMOTA   0.477
AMOTP   1.136
RECALL  0.588
MOTAR   0.726
GT  14556
MOTA    0.437
MOTP    0.658
MT  3115
ML  1963
FAF 51.4
TP  68969
FP  14113
FN  32487
IDS 441
FRAG    628
TID 1.37
LGD 1.89
"""

# ================ base config ===================
# 是否启用自定义插件目录（确保框架能加载projects下的插件模块）
plugin = True
# 自定义插件代码所在的根目录
plugin_dir = "projects/mmdet3d_plugin/"
# 分布式训练通信后端配置
dist_params = dict(backend="nccl")
# 日志输出等级
log_level = "INFO"
# mmcv默认工作目录，可在命令行覆盖
work_dir = None

# 单个迭代的总batch size = 每块GPU的batch size * GPU数量
total_batch_size = 48
# 当前配置期望使用的GPU数量
num_gpus = 8
# 每张GPU上实际使用的batch size
batch_size = total_batch_size // num_gpus
# 每个epoch包含的迭代次数（按nuScenes训练集样本数估算）
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
# 训练总轮数
num_epochs = 100
# 保存checkpoint的epoch间隔
checkpoint_epoch_interval = 20

checkpoint_config = dict(
    # checkpoint保存间隔（按迭代数指定）
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    # 多少个iter打印一次日志
    interval=51,
    hooks=[
        # 文本日志记录器
        dict(type="TextLoggerHook", by_epoch=False),
        # TensorBoard日志记录器
        dict(type="TensorboardLoggerHook"),
    ],
)
# 预训练权重文件路径，默认不加载
load_from = None
# 断点续训路径，默认从头训练
resume_from = None
# 训练流程定义，单阶段训练
workflow = [("train", 1)]
# 混合精度训练配置
fp16 = dict(loss_scale=32.0)  # 静态loss缩放因子
# 输入图像的高宽（W,H），用于数据增强与网络预设
input_shape = (704, 256)

# 是否在测试阶段启用跟踪推理
tracking_test = True
# 跟踪时的置信度阈值
tracking_threshold = 0.2

# ================== model ========================
# 模型需要预测的类别名称
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# 类别数量
num_classes = len(class_names)
# 视觉特征通道维度（Transformer嵌入维度）
embed_dims = 256
# 多头注意力的分组数/头数
num_groups = 8
# 解码层总层数
num_decoder = 6
# 单帧解码层数量（其余为时序解码）
num_single_frame_decoder = 1
# 是否使用自定义可变形采样算子（需提前编译）
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
# FPN输出特征图对应的下采样步长
strides = [4, 8, 16, 32]
# FPN尺度层数
num_levels = len(strides)
# 深度分支的尺度层数
num_depth_layers = 3
# Transformer中使用的dropout比率
drop_out = 0.1
# 是否启用时序建模
temporal = True
# 是否使用解耦注意力结构
decouple_attn = True
# 是否输出质量估计分支（用于跟踪等任务）
with_quality_estimation = True

model = dict(
    # 模型主体类型
    type="Sparse4D",
    # 是否使用GridMask数据增强
    use_grid_mask=True,
    # 是否启用自定义可变形算子
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        # 图像骨干网络类型
        type="ResNet",
        # ResNet深度
        depth=50,
        # 网络阶段数量
        num_stages=4,
        # 冻结前几层，-1表示全部可训练
        frozen_stages=-1,
        # eval模式下是否冻结BN统计量
        norm_eval=False,
        # ResNet风格
        style="pytorch",
        # 是否使用checkpoint以节省显存
        with_cp=True,
        # 输出的特征层索引
        out_indices=(0, 1, 2, 3),
        # BN归一化配置
        norm_cfg=dict(type="BN", requires_grad=True),
        # 预训练权重路径
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        # 颈部结构使用FPN
        type="FPN",
        # 输出特征层数量
        num_outs=num_levels,
        # 起始输入层索引
        start_level=0,
        # 输出通道数
        out_channels=embed_dims,
        # 是否在输出上额外添加卷积
        add_extra_convs="on_output",
        # 在额外卷积之前先ReLU
        relu_before_extra_convs=True,
        # 各输入特征层通道数
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        # 深度预测辅助分支
        type="DenseDepthNet",
        # 深度分支的嵌入维度
        embed_dims=embed_dims,
        # 深度预测的尺度层数
        num_depth_layers=num_depth_layers,
        # 辅助损失权重
        loss_weight=0.2,
    ),
    head=dict(
        # Sparse4D检测头
        type="Sparse4DHead",
        # 分类分数超过该阈值才进入回归阶段
        cls_threshold_to_reg=0.05,
        # 是否使用解耦注意力
        decouple_attn=decouple_attn,
        instance_bank=dict(
            # 实例缓存模块
            type="InstanceBank",
            # Anchor数量
            num_anchor=900,
            # Anchor特征维度
            embed_dims=embed_dims,
            # Anchor初始位置文件
            anchor="nuscenes_kmeans900.npy",
            # Anchor生成器配置
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            # 时序实例缓存数量
            num_temp_instances=600 if temporal else -1,
            # 时序置信度衰减系数
            confidence_decay=0.6,
            # 是否对缓存特征反向传播
            feat_grad=False,
        ),
        anchor_encoder=dict(
            # Anchor编码器配置
            type="SparseBox3DEncoder",
            # 速度向量维度
            vel_dims=3,
            # 编码后嵌入维度设置
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            # 多模态融合方式
            mode="cat" if decouple_attn else "add",
            # 是否输出额外全连接层
            output_fc=not decouple_attn,
            # 输入循环次数
            in_loops=1,
            # 输出循环次数
            out_loops=4 if decouple_attn else 2,
        ),
        # 单帧解码层数量
        num_single_frame_decoder=num_single_frame_decoder,
        # 每层操作顺序定义
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],  # 前两步为历史保留项，截断保持与实现对齐
        temp_graph_model=dict(
            # 时序关系建模的注意力模块
            type="MultiheadAttention",
            # 注意力输入维度
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            # 注意力头数
            num_heads=num_groups,
            # 使用 batch_first 形式
            batch_first=True,
            # Dropout比例
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            # 空间关系建模的注意力模块
            type="MultiheadAttention",
            # 注意力输入维度
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            # 注意力头数
            num_heads=num_groups,
            # 使用 batch_first 形式
            batch_first=True,
            # Dropout比例
            dropout=drop_out,
        ),
        # LayerNorm配置
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            # 前馈网络结构
            type="AsymmetricFFN",
            # 输入通道数
            in_channels=embed_dims * 2,
            # 前置归一化配置
            pre_norm=dict(type="LN"),
            # 输出嵌入维度
            embed_dims=embed_dims,
            # 隐藏层通道数
            feedforward_channels=embed_dims * 4,
            # 全连接层数量
            num_fcs=2,
            # FFN的dropout比例
            ffn_drop=drop_out,
            # 激活函数设置
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            # 多视角特征聚合模块
            type="DeformableFeatureAggregation",
            # 输入嵌入维度
            embed_dims=embed_dims,
            # 分组数量
            num_groups=num_groups,
            # 使用的特征层数量
            num_levels=num_levels,
            # 摄像头数量
            num_cams=6,
            # 注意力dropout
            attn_drop=0.15,
            # 是否使用自定义可变形算子
            use_deformable_func=use_deformable_func,
            # 是否使用相机嵌入编码
            use_camera_embed=True,
            # 残差连接的拼接模式
            residual_mode="cat",
            kps_generator=dict(
                # 关键点生成器
                type="SparseBox3DKeyPointsGenerator",
                # 可学习的关键点数量
                num_learnable_pts=6,
                # 固定关键点相对尺度
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            # 3D框精修模块
            type="SparseBox3DRefinementModule",
            # 精修层的嵌入维度
            embed_dims=embed_dims,
            # 分类数量
            num_cls=num_classes,
            # 是否优化偏航角
            refine_yaw=True,
            # 是否输出质量估计
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            # 匹配与降噪采样器
            type="SparseBox3DTarget",
            # 去噪组数量
            num_dn_groups=5,
            # 时序去噪组数量
            num_temp_dn_groups=3,
            # 去噪扰动尺度（位置/尺寸等）
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            # 每个batch最多加入的去噪GT数量
            max_dn_gt=32,
            # 是否添加负样本去噪
            add_neg_dn=True,
            # 分类损失权重
            cls_weight=2.0,
            # 框回归损失权重
            box_weight=0.25,
            # 回归损失各分量权重
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            # 针对特定类别设置的回归权重
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            # 分类损失配置
            type="FocalLoss",
            # 是否使用sigmoid形式
            use_sigmoid=True,
            # Focal Loss的gamma参数
            gamma=2.0,
            # Focal Loss的alpha参数
            alpha=0.25,
            # 分类损失权重
            loss_weight=2.0,
        ),
        loss_reg=dict(
            # 回归损失配置
            type="SparseBox3DLoss",
            # 3D框回归的L1损失
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            # 框中心度损失
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            # 偏航角分布损失
            loss_yawness=dict(type="GaussianFocalLoss"),
            # 允许偏航角翻转的类别索引
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        # 解码器结构类型
        decoder=dict(type="SparseBox3DDecoder"),
        # 损失中回归项权重
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
# 使用的dataset实现类
dataset_type = "NuScenes3DDetTrackDataset"
# 原始nuScenes数据根目录
data_root = "data/nuscenes/"
# 相机标注目录（旧变量，保留以兼容）
anno_root = "data/nuscenes_cam/"
# 实际使用的标注pkl目录
anno_root = "data/nuscenes_anno_pkls/"
# 文件读取后端配置
file_client_args = dict(backend="disk")

# 图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 图像均值
    std=[58.395, 57.12, 57.375],  # 图像标准差
    to_rgb=True,  # 是否将BGR转换为RGB
)
# 训练数据处理流水线
train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",  # 读取多视角相机图像
        to_float32=True,  # 将像素转换为float32以适配后续算子
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",  # 点云坐标系类型
        load_dim=5,  # 从文件读取的点特征维度
        use_dim=5,  # 实际送入模型的点特征维度
        file_client_args=file_client_args,  # 点云读取后端
    ),
    dict(
        type="ResizeCropFlipImage",  # 多尺度缩放/裁剪/翻转增强
    ),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],  # 深度图下采样比例列表
    ),
    dict(type="BBoxRotation"),  # 随机旋转3D框
    dict(type="PhotoMetricDistortionMultiViewImage"),  # 光照颜色扰动
    dict(
        type="NormalizeMultiviewImage",
        **img_norm_cfg,
    ),  # 图像归一化
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),  # 各类别的半径过滤阈值
    ),
    dict(
        type="InstanceNameFilter",
        classes=class_names,  # 仅保留指定类别
    ),
    dict(type="NuScenesSparse4DAdaptor"),  # 转换为Sparse4D所需输入格式
    dict(
        type="Collect",
        keys=[
            "img",  # 经过增强的多视角图像
            "timestamp",  # 当前帧时间戳
            "projection_mat",  # 相机外参到图像的投影矩阵
            "image_wh",  # 图像宽高信息
            "gt_depth",  # 监督使用的深度图
            "focal",  # 相机内参焦距
            "gt_bboxes_3d",  # 3D真实框
            "gt_labels_3d",  # 3D框对应的类别
        ],
        meta_keys=[
            "T_global",  # 帧到全局坐标变换
            "T_global_inv",  # 全局到帧的逆变换
            "timestamp",  # 序列对齐所需时间戳
            "instance_id",  # 实例ID用于跟踪
        ],
    ),
]
# 测试/验证数据处理流水线
test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",  # 读取测试图像
        to_float32=True,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",  # 多视角图像
            "timestamp",  # 当前帧时间戳
            "projection_mat",  # 投影矩阵
            "image_wh",  # 图像尺寸
        ],
        meta_keys=[
            "T_global",  # 帧到全局位姿
            "T_global_inv",  # 全局到帧位姿
            "timestamp",  # 用于结果排序的时间戳
        ],  # 评估时需要的元数据
    ),
]

input_modality = dict(
    use_lidar=False,  # 不使用激光雷达点云
    use_camera=True,  # 使用多视角相机图像
    use_radar=False,  # 不使用雷达数据
    use_map=False,  # 不加载地图先验
    use_external=False,  # 不额外使用外部数据
)

data_basic_config = dict(
    # 基础数据集配置，供train/val/test复用
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),  # 图像缩放比例范围
    "final_dim": input_shape[::-1],  # 最终尺寸（H, W）
    "bot_pct_lim": (0.0, 0.0),  # 底部裁剪比例范围
    "rot_lim": (-5.4, 5.4),  # 图像旋转角范围
    "H": 900,  # 原始图像高度
    "W": 1600,  # 原始图像宽度
    "rand_flip": True,  # 是否随机翻转
    "rot3d_range": [-0.3925, 0.3925],  # 3D增强旋转范围（弧度）
}

data = dict(
    # 每个GPU上的样本数
    samples_per_gpu=batch_size,
    # 每个GPU上的dataloader工作线程数
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        # 训练集标注文件
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        # 训练使用的pipeline
        pipeline=train_pipeline,
        # 是否测试模式
        test_mode=False,
        # 传入相机增强配置
        data_aug_conf=data_aug_conf,
        # 是否按序列方式加载
        with_seq_flag=True,
        # 将序列拆分的数量
        sequences_split_num=2,
        # 是否在序列内保持一致的增强
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",  # 验证集标注
        pipeline=test_pipeline,  # 验证阶段的数据流水线
        data_aug_conf=data_aug_conf,  # 评估时使用同样的相机增强参数
        test_mode=True,  # 以测试模式禁用训练增强
        tracking=tracking_test,  # 是否在验证时输出跟踪结果
        tracking_threshold=tracking_threshold,  # 跟踪分数阈值
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",  # 测试集标注（默认使用验证集）
        pipeline=test_pipeline,  # 测试阶段的数据流水线
        data_aug_conf=data_aug_conf,  # 相机增强参数
        test_mode=True,  # 启用测试模式
        tracking=tracking_test,  # 是否输出跟踪
        tracking_threshold=tracking_threshold,  # 跟踪置信度阈值
    ),
)

# ================== training ========================
optimizer = dict(
    # 优化器类型
    type="AdamW",
    # 初始学习率
    lr=6e-4,
    # 权重衰减系数
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            # 针对图像骨干设置较小学习率
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
# 梯度裁剪配置
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    # 学习率策略：余弦退火
    policy="CosineAnnealing",
    # 预热策略
    warmup="linear",
    # 预热迭代次数
    warmup_iters=500,
    # 预热初始比例
    warmup_ratio=1.0 / 3,
    # 最低学习率比例
    min_lr_ratio=1e-3,
)
runner = dict(
    # 迭代式训练器
    type="IterBasedRunner",
    # 总迭代数 = 每epoch迭代 * epoch数
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",  # 读取图像用于可视化
        to_float32=True,  # 转换数据类型
    ),
    dict(
        type="Collect",
        keys=["img"],  # 仅保留图像用于可视化
        meta_keys=["timestamp", "lidar2img"],  # 可视化需求的元信息
    ),
]
evaluation = dict(
    # 验证间隔（以迭代数为单位）
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    # 验证时可选的可视化pipeline
    pipeline=vis_pipeline,
    # out_dir="./vis",  # for visualization
)
