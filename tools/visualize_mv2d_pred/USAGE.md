# MV2DFusion可视化工具使用说明

## 实现完成情况

✅ **已完成的功能：**
- MV2DFusion数据格式解析和加载
- 3D检测框格式转换
- lidar2img变换矩阵提取
- 模拟预测结果生成（用于测试）
- 核心可视化逻辑实现
- 多相机图像布局设计
- 分数阈值过滤

✅ **测试验证：**
- 数据格式分析测试通过
- 核心逻辑流水线测试通过
- lidar2img矩阵提取测试通过

## 文件说明

```
tools/visualize_mv2d_pred/
├── mv2dfusion_visualizer.py  # 主要可视化类（完整版本）
├── example_usage.py          # 使用示例脚本
├── simple_test.py            # 简化测试脚本（验证核心逻辑）
├── test_basic.py             # 基础数据格式测试
├── README.md                 # 详细说明文档
└── USAGE.md                  # 本使用说明
```

## 核心实现说明

### 1. 数据格式适配

已成功解析MV2DFusion数据格式：
- 支持GT框格式：`[x, y, z, w, l, h, yaw]`
- 支持多相机数据：`CAM_FRONT`, `CAM_FRONT_RIGHT`, 等
- 提取相机内外参：`intrinsics`, `ego2cam`
- 支持16个百度数据类别

### 2. 类别定义更新

已更新为百度数据的完整类别定义：
```python
CLASSES = [
    "car", "truck", "bus", "engineering_vehicle", "cyclist", "person", 
    "traffic_cone", "tripod", "tricycle", "water_horse", "pillar", 
    "noparking_pillar", "gate_on", "gate_off", "anti_collision_bucket", 
    "noparking_board"
]
```

每个类别都有对应的颜色映射，总共16个类别。

### 3. save_pred函数修改要点

基于原始`save_pred`函数，主要修改：

```python
def save_pred(self, pts, imgs_pure, pred_box, lidar2img, 
              gt_bboxes_3d, gt_labels_3d, save_dir, every_frame=False):
    # 关键修改点：
    # 1. vis_info=None (MV2DFusion没有instance_ids)
    # 2. 适配MV2DFusion的预测格式
    # 3. 保持原有的绘制函数调用
    # 4. 维持相同的可视化效果
```

### 4. 调用接口

```python
from mv2dfusion_visualizer import MV2DFusionVisualizer

# 创建可视化器
visualizer = MV2DFusionVisualizer(vis_score_threshold=0.3)

# 加载数据
data = visualizer.load_data("data.pkl")
predictions = visualizer.load_data("predictions.pkl")

# 可视化
visualizer.visualize_frame(data, predictions, frame_idx=0, save_dir="./output")
```

## 环境依赖处理

代码已处理常见依赖问题：
- **torch不可用**：使用numpy替代
- **cv2不可用**：创建占位图像
- **绘制函数缺失**：导入原项目工具函数

## 测试结果

```bash
# 运行简化测试
python simple_test.py

# 输出示例：
============================================================
测试MV2DFusion可视化流水线
============================================================

1. 加载数据...
   加载完成，共 2 帧

2. 获取第 0 帧信息...
   GT框数量: 18
   相机数量: 11

3. 提取lidar2img变换矩阵...
   提取完成，共 7 个矩阵

4. 创建模拟预测结果...
   生成 3 个预测结果

5. 过滤低分数预测...
   过滤前: 3 个预测
   过滤后: 3 个预测

6. 提取GT数据...
   GT框: (18, 7)
   GT标签: (18,)

7. 可视化准备完成!
   预测框数量: 3
   GT框数量: 18
   相机数量: 7

测试完成！核心逻辑验证成功
```

## 下一步操作

要完整运行可视化，需要：

1. **安装依赖**：
   ```bash
   pip install torch opencv-python
   ```

2. **准备预测结果**：
   - 将MV2DFusion模型推理结果转换为指定格式
   - 或使用内置的模拟预测功能测试

3. **运行可视化**：
   ```bash
   python example_usage.py \
       --data_path /path/to/mv2dfusion_data.pkl \
       --pred_path /path/to/predictions.pkl \
       --save_dir ./output
   ```

## 关键特点

1. **完全基于原save_pred函数**：保持相同的可视化效果
2. **最小化代码修改**：只修改数据输入格式，不改动绘制逻辑
3. **灵活的依赖处理**：即使缺少某些库也能运行核心逻辑
4. **vis_info=None**：按要求不显示instance_ids
5. **模块化设计**：易于扩展和维护

代码实现已完成，核心逻辑测试通过，可以开始进行实际的可视化测试！