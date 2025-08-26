#!/usr/bin/env python3
"""
基础测试脚本 - 验证数据加载和格式转换
"""

import os
import sys
import json
import pickle
import numpy as np

def load_data(data_path):
    """加载数据文件"""
    if data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    return data

def analyze_mv2d_data(data_path):
    """分析MV2DFusion数据格式"""
    print(f"加载数据: {data_path}")
    data = load_data(data_path)
    
    print(f"数据键: {list(data.keys())}")
    print(f"帧数: {len(data['infos'])}")
    
    # 分析第一帧
    if len(data['infos']) > 0:
        frame_info = data['infos'][0]
        print(f"\n第一帧信息:")
        print(f"  键: {list(frame_info.keys())}")
        
        if 'gt_boxes' in frame_info:
            gt_boxes = np.array(frame_info['gt_boxes'])
            print(f"  GT框数量: {len(gt_boxes)}")
            if len(gt_boxes) > 0:
                print(f"  GT框形状: {gt_boxes.shape}")
                print(f"  第一个GT框: {gt_boxes[0]}")
        
        if 'gt_labels_3d' in frame_info:
            gt_labels = np.array(frame_info['gt_labels_3d'])
            print(f"  GT标签: {gt_labels}")
        
        if 'class_name' in frame_info:
            class_names = frame_info['class_name']
            print(f"  类别名称: {class_names}")
        
        if 'cams' in frame_info:
            cams = frame_info['cams']
            print(f"  相机数量: {len(cams)}")
            print(f"  相机名称: {list(cams.keys())}")
            
            # 分析第一个相机
            first_cam = list(cams.keys())[0]
            cam_info = cams[first_cam]
            print(f"  {first_cam} 信息:")
            print(f"    键: {list(cam_info.keys())}")
            if 'intrinsics' in cam_info:
                intrinsics = np.array(cam_info['intrinsics'])
                print(f"    内参矩阵形状: {intrinsics.shape}")
            if 'ego2cam' in cam_info:
                ego2cam = np.array(cam_info['ego2cam'])
                print(f"    ego2cam矩阵形状: {ego2cam.shape}")

def create_mock_predictions(data, frame_idx):
    """创建模拟预测结果"""
    frame_info = data['infos'][frame_idx]
    gt_boxes = np.array(frame_info['gt_boxes'])
    gt_labels = np.array(frame_info['gt_labels_3d'])
    
    if len(gt_boxes) > 0:
        # 选择前几个GT作为预测
        num_preds = min(3, len(gt_boxes))
        pred_boxes = gt_boxes[:num_preds].copy()
        pred_labels = gt_labels[:num_preds].copy()
        
        # 添加噪声
        pred_boxes[:, :3] += np.random.normal(0, 0.5, (num_preds, 3))
        pred_scores = np.random.uniform(0.4, 0.9, num_preds)
        
        predictions = {
            'boxes_3d': pred_boxes.tolist(),
            'scores_3d': pred_scores.tolist(),
            'labels_3d': pred_labels.tolist(),
            'cls_scores': pred_scores.tolist()
        }
    else:
        predictions = {
            'boxes_3d': [],
            'scores_3d': [],
            'labels_3d': [],
            'cls_scores': []
        }
    
    return predictions

def test_format_conversion():
    """测试格式转换"""
    data_path = "/home/yr/yr/code/cv/object_detection/3D_OD/MV2DFusion-gly/data/Baidu_OD/train_pkl/test_data.json"
    
    print("=" * 50)
    print("测试MV2DFusion数据格式分析")
    print("=" * 50)
    
    analyze_mv2d_data(data_path)
    
    print("\n" + "=" * 50)
    print("测试模拟预测结果生成")
    print("=" * 50)
    
    data = load_data(data_path)
    predictions = create_mock_predictions(data, 0)
    
    print(f"预测结果:")
    for key, value in predictions.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} 个元素")
            if len(value) > 0:
                print(f"    第一个: {value[0]}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_format_conversion()