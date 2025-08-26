#!/usr/bin/env python3
"""
MV2DFusion可视化示例脚本

使用方法:
python example_usage.py --data_path /path/to/data.pkl --pred_path /path/to/predictions.pkl --save_dir /path/to/output

"""

import argparse
import os
import sys
import numpy as np

from mv2dfusion_visualizer import MV2DFusionVisualizer


def create_mock_predictions(data, frame_idx):
    """
    创建模拟预测结果，用于测试
    
    Args:
        data: 数据字典
        frame_idx: 帧索引
        
    Returns:
        模拟的预测结果
    """
    frame_info = data['infos'][frame_idx]
    gt_boxes = np.array(frame_info['gt_boxes'])
    gt_labels = np.array(frame_info['gt_labels_3d'])
    
    # 基于GT创建一些模拟预测（添加一些噪声）
    if len(gt_boxes) > 0:
        # 随机选择一些GT框作为预测结果
        num_preds = min(len(gt_boxes), np.random.randint(1, len(gt_boxes) + 1))
        indices = np.random.choice(len(gt_boxes), num_preds, replace=False)
        
        pred_boxes = gt_boxes[indices].copy()
        pred_labels = gt_labels[indices].copy()
        
        # 添加一些噪声
        pred_boxes[:, :3] += np.random.normal(0, 0.5, (num_preds, 3))  # 位置噪声
        pred_boxes[:, 3:6] += np.random.normal(0, 0.1, (num_preds, 3))  # 尺寸噪声
        pred_boxes[:, 6] += np.random.normal(0, 0.1, num_preds)  # 角度噪声
        
        # 生成随机分数
        pred_scores = np.random.uniform(0.3, 0.9, num_preds)
        
        predictions = {
            'boxes_3d': pred_boxes,
            'scores_3d': pred_scores,
            'labels_3d': pred_labels,
            'cls_scores': pred_scores
        }
    else:
        predictions = {
            'boxes_3d': np.empty((0, 7)),
            'scores_3d': np.empty(0),
            'labels_3d': np.empty(0, dtype=int),
            'cls_scores': np.empty(0)
        }
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='MV2DFusion结果可视化')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='MV2DFusion数据文件路径(.pkl或.json)')
    parser.add_argument('--pred_path', type=str, default=None,
                       help='预测结果文件路径(.pkl)')
    parser.add_argument('--save_dir', type=str, default='./visualization_output',
                       help='可视化结果保存目录')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='要可视化的帧索引')
    parser.add_argument('--vis_score_threshold', type=float, default=0.3,
                       help='可视化分数阈值')
    parser.add_argument('--every_frame', action='store_true',
                       help='是否按帧保存图像')
    parser.add_argument('--use_mock_pred', action='store_true',
                       help='使用模拟预测结果（用于测试）')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = MV2DFusionVisualizer(vis_score_threshold=args.vis_score_threshold)
    
    # 加载数据
    print(f"加载数据: {args.data_path}")
    data = visualizer.load_data(args.data_path)
    print(f"数据加载完成，共有 {len(data['infos'])} 帧")
    
    # 检查帧索引
    if args.frame_idx >= len(data['infos']):
        print(f"错误: 帧索引 {args.frame_idx} 超出范围 [0, {len(data['infos'])-1}]")
        return
    
    # 加载或创建预测结果
    if args.use_mock_pred or args.pred_path is None:
        print("使用模拟预测结果")
        predictions = create_mock_predictions(data, args.frame_idx)
    else:
        print(f"加载预测结果: {args.pred_path}")
        predictions = visualizer.load_data(args.pred_path)
    
    # 可视化指定帧
    print(f"可视化第 {args.frame_idx} 帧")
    visualizer.visualize_frame(
        data=data,
        predictions=predictions,
        frame_idx=args.frame_idx,
        save_dir=args.save_dir,
        every_frame=args.every_frame
    )
    
    print(f"可视化完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()