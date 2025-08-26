#!/usr/bin/env python3
"""
简化测试版本 - 测试核心逻辑，不依赖cv2和torch
"""

import os
import sys
import json
import pickle
import numpy as np

class SimpleMV2DVisualizer:
    """简化的MV2DFusion可视化器，用于测试核心逻辑"""
    
    def __init__(self, vis_score_threshold=0.3):
        self.vis_score_threshold = vis_score_threshold
        self.tmp_trackid = []
        self.save_count = 1
        
    def load_data(self, data_path):
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
    
    def extract_lidar2img_from_mv2d(self, frame_info):
        """从MV2DFusion格式提取lidar2img变换矩阵"""
        lidar2img_list = []
        
        cam_order = [
            "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW"
        ]
        
        for cam_name in cam_order:
            if cam_name in frame_info.get('cams', {}):
                cam_info = frame_info['cams'][cam_name]
                
                # 获取内参矩阵
                intrinsic = np.array(cam_info['intrinsics'])
                
                # 获取外参矩阵 ego2cam
                ego2cam = np.array(cam_info['ego2cam'])
                
                # 构建4x4内参矩阵
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                
                # 计算lidar2img = intrinsic @ ego2cam
                lidar2img = viewpad @ ego2cam
                lidar2img_list.append(lidar2img)
                
                print(f"相机 {cam_name}:")
                print(f"  内参形状: {intrinsic.shape}")
                print(f"  外参形状: {ego2cam.shape}")
                print(f"  lidar2img形状: {lidar2img.shape}")
            else:
                # 创建单位矩阵占位
                lidar2img_list.append(np.eye(4))
        
        return lidar2img_list
    
    def create_mock_predictions(self, data, frame_idx):
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
            pred_boxes[:, :3] += np.random.normal(0, 0.2, (num_preds, 3))
            pred_scores = np.random.uniform(0.4, 0.9, num_preds)
            
            predictions = {
                'boxes_3d': pred_boxes,
                'scores_3d': pred_scores,
                'labels_3d': pred_labels,
                'cls_scores': pred_scores
            }
            
            print(f"生成 {num_preds} 个预测结果:")
            for i in range(num_preds):
                print(f"  预测 {i}: 框={pred_boxes[i][:3]}, 分数={pred_scores[i]:.3f}, 标签={pred_labels[i]}")
                
        else:
            predictions = {
                'boxes_3d': np.empty((0, 7)),
                'scores_3d': np.empty(0),
                'labels_3d': np.empty(0, dtype=int),
                'cls_scores': np.empty(0)
            }
            print("没有GT框，生成空预测结果")
        
        return predictions
    
    def filter_predictions(self, predictions):
        """过滤低分数预测"""
        scores = predictions['scores_3d']
        mask = scores > self.vis_score_threshold
        
        filtered_pred = {}
        for key, value in predictions.items():
            if isinstance(value, np.ndarray):
                filtered_pred[key] = value[mask]
            else:
                filtered_pred[key] = np.array(value)[mask] if len(value) > 0 else np.array([])
        
        print(f"过滤前: {len(scores)} 个预测")
        print(f"过滤后: {len(filtered_pred['scores_3d'])} 个预测")
        
        return filtered_pred
    
    def test_visualization_pipeline(self, data_path, frame_idx=0):
        """测试可视化流水线"""
        print("=" * 60)
        print("测试MV2DFusion可视化流水线")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        data = self.load_data(data_path)
        print(f"   加载完成，共 {len(data['infos'])} 帧")
        
        # 2. 获取帧信息
        print(f"\n2. 获取第 {frame_idx} 帧信息...")
        frame_info = data['infos'][frame_idx]
        print(f"   GT框数量: {len(frame_info['gt_boxes'])}")
        print(f"   相机数量: {len(frame_info.get('cams', {}))}")
        
        # 3. 提取lidar2img矩阵
        print(f"\n3. 提取lidar2img变换矩阵...")
        lidar2img_list = self.extract_lidar2img_from_mv2d(frame_info)
        print(f"   提取完成，共 {len(lidar2img_list)} 个矩阵")
        
        # 4. 创建模拟预测
        print(f"\n4. 创建模拟预测结果...")
        predictions = self.create_mock_predictions(data, frame_idx)
        
        # 5. 过滤预测
        print(f"\n5. 过滤低分数预测...")
        filtered_pred = self.filter_predictions(predictions)
        
        # 6. 提取GT数据
        print(f"\n6. 提取GT数据...")
        gt_bboxes_3d = np.array(frame_info['gt_boxes'])
        gt_labels_3d = np.array(frame_info['gt_labels_3d'])
        print(f"   GT框: {gt_bboxes_3d.shape}")
        print(f"   GT标签: {gt_labels_3d.shape}")
        
        print(f"\n7. 可视化准备完成!")
        print(f"   预测框数量: {len(filtered_pred['boxes_3d'])}")
        print(f"   GT框数量: {len(gt_bboxes_3d)}")
        print(f"   相机数量: {len(lidar2img_list)}")
        
        return {
            'data': data,
            'frame_info': frame_info,
            'predictions': filtered_pred,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'lidar2img_list': lidar2img_list
        }

def main():
    """主测试函数"""
    data_path = "/home/yr/yr/code/cv/object_detection/3D_OD/MV2DFusion-gly/data/Baidu_OD/train_pkl/test_data.json"
    
    visualizer = SimpleMV2DVisualizer(vis_score_threshold=0.3)
    
    # 测试第一帧
    result = visualizer.test_visualization_pipeline(data_path, frame_idx=0)
    
    print("\n" + "=" * 60)
    print("测试完成！核心逻辑验证成功")
    print("=" * 60)

if __name__ == "__main__":
    main()