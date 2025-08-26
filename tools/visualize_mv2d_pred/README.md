# MV2DFusion模型推理结果可视化

这个工具用于可视化MV2DFusion模型的推理结果，将3D检测框投影到多相机图像和点云BEV视图上。

## 文件结构

```
tools/visualize_mv2d_pred/
├── mv2dfusion_visualizer.py  # 主要的可视化类
├── example_usage.py          # 使用示例脚本
└── README.md                 # 说明文档
```

## 功能特点

- 支持MV2DFusion数据格式（.pkl和.json）
- 将3D检测框投影到多相机图像上
- 生成点云BEV视图可视化
- 支持GT和预测结果同时显示
- 灵活的分数阈值控制
- 支持单帧或批量可视化

## 使用方法

### 1. 基本使用

```bash
# 使用模拟预测结果测试
python example_usage.py \
    --data_path /path/to/mv2dfusion_data.pkl \
    --save_dir ./output \
    --use_mock_pred

# 使用真实预测结果
python example_usage.py \
    --data_path /path/to/mv2dfusion_data.pkl \
    --pred_path /path/to/predictions.pkl \
    --save_dir ./output
```

### 2. 参数说明

- `--data_path`: MV2DFusion数据文件路径（支持.pkl和.json）
- `--pred_path`: 预测结果文件路径（.pkl格式）
- `--save_dir`: 可视化结果保存目录
- `--frame_idx`: 要可视化的帧索引（默认为0）
- `--vis_score_threshold`: 可视化分数阈值（默认0.3）
- `--every_frame`: 是否按帧保存图像
- `--use_mock_pred`: 使用模拟预测结果（用于测试）

### 3. 编程接口使用

```python
from mv2dfusion_visualizer import MV2DFusionVisualizer

# 创建可视化器
visualizer = MV2DFusionVisualizer(vis_score_threshold=0.3)

# 加载数据
data = visualizer.load_data("data.pkl")
predictions = visualizer.load_data("predictions.pkl")

# 可视化单帧
visualizer.visualize_frame(
    data=data,
    predictions=predictions,
    frame_idx=0,
    save_dir="./output"
)
```

## 数据格式要求

### 输入数据格式（MV2DFusion）

```python
{
    "metadata": {...},
    "infos": [
        {
            "lidar_path_360_ego": "path/to/lidar.bin",
            "gt_boxes": [[x, y, z, w, l, h, yaw], ...],
            "gt_labels_3d": [0, 1, 2, ...],
            "class_name": ["car", "truck", ...],
            "cams": {
                "CAM_FRONT": {
                    "data_path": "path/to/image.jpg",
                    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    "ego2cam": [[...], [...], [...], [...]]
                },
                ...
            }
        },
        ...
    ]
}
```

### 预测结果格式

```python
{
    "boxes_3d": [[x, y, z, w, l, h, yaw], ...],  # 3D检测框
    "scores_3d": [0.8, 0.7, ...],               # 置信度分数
    "labels_3d": [0, 1, ...],                   # 类别标签
    "cls_scores": [0.8, 0.7, ...]               # 分类分数
}
```

## 类别定义

支持以下类别（与百度数据格式对应）：

| ID | 类别名称 | 颜色 |
|----|----------|------|
| 0  | car | 蓝色 |
| 1  | truck | 绿色 |
| 2  | bus | 红色 |
| 3  | engineering_vehicle | 黄色 |
| 4  | cyclist | 青色 |
| 5  | person | 紫色 |
| 6  | traffic_cone | 淡紫色 |
| 7  | tripod | 橙蓝色 |
| 8  | tricycle | 浅蓝色 |
| 9  | water_horse | 橄榄色 |
| 10 | pillar | 浅紫色 |
| 11 | noparking_pillar | 深青色 |
| 12 | gate_on | 深绿色 |
| 13 | gate_off | 深红色 |
| 14 | anti_collision_bucket | 橙色 |
| 15 | noparking_board | 紫色 |

## 输出结果

可视化结果包含：

1. **多相机视图**: 7个相机的图像，显示投影的3D检测框
2. **BEV视图**: 点云鸟瞰图，显示3D检测框
3. **拼接图像**: 将所有视图拼接成一张大图

图像布局：
```
    [ ]  [front_narrow]  [ ]
[left] [front] [right]
[rear_left] [rear] [rear_right]
```

## 注意事项

1. 确保点云文件路径正确且文件存在
2. 确保相机图像路径正确且文件存在
3. 如果某些文件不存在，会使用空白占位图像
4. MV2DFusion没有instance_ids，所以不显示跟踪ID
5. 分数阈值用于过滤低置信度的预测结果

## 依赖项

- OpenCV (cv2)
- NumPy
- PyTorch
- 项目内的工具函数 (draw_lidar_bbox3d_on_img, draw_lidar_bbox3d_points_on_bev)

## 故障排除

1. **"模块导入错误"**: 确保项目路径已添加到sys.path
2. **"文件不存在"**: 检查数据文件路径是否正确
3. **"可视化结果为空"**: 检查分数阈值是否过高
4. **"图像显示异常"**: 检查相机内外参数是否正确