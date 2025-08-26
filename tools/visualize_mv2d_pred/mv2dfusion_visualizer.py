import os
import sys
import json
import pickle
import numpy as np
import cv2
from typing import Dict, List, Any

# 添加项目路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# 尝试导入torch，如果失败则创建简单的替代
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 创建简单的torch替代类
    class torch:
        @staticmethod
        def tensor(data):
            return np.array(data)
    
    class TensorLike:
        def __init__(self, data):
            self.data = np.array(data)
        
        def cpu(self):
            return self
        
        def numpy(self):
            return self.data

from projects.mmdet3d_plugin.datasets.utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_points_on_bev
)


class MV2DFusionVisualizer:
    """MV2DFusion模型推理结果可视化器"""
    
    # 类别定义，与百度数据格式对应
    CLASSES = [
        "car",                    # 0
        "truck",                  # 1  
        "bus",                    # 2
        "engineering_vehicle",    # 3
        "cyclist",                # 4
        "person",                 # 5
        "traffic_cone",          # 6
        "tripod",                # 7
        "tricycle",              # 8
        "water_horse",           # 9
        "pillar",                # 10
        "noparking_pillar",      # 11
        "gate_on",               # 12
        "gate_off",              # 13
        "anti_collision_bucket", # 14
        "noparking_board"        # 15
    ]
    
    # 颜色映射
    ID_COLOR_MAP = [
        (59, 59, 238),      # 0 - car - 蓝色
        (0, 255, 0),        # 1 - truck - 绿色
        (0, 0, 255),        # 2 - bus - 红色
        (255, 255, 0),      # 3 - engineering_vehicle - 黄色
        (0, 255, 255),      # 4 - cyclist - 青色
        (255, 0, 255),      # 5 - person - 紫色
        (255, 128, 255),    # 6 - traffic_cone - 淡紫色
        (0, 127, 255),      # 7 - tripod - 橙蓝色
        (71, 130, 255),     # 8 - tricycle - 浅蓝色
        (127, 127, 0),      # 9 - water_horse - 橄榄色
        (128, 128, 255),    # 10 - pillar - 浅紫色
        (0, 127, 127),      # 11 - noparking_pillar - 深青色
        (71, 127, 100),     # 12 - gate_on - 深绿色
        (127, 34, 34),      # 13 - gate_off - 深红色
        (255, 165, 0),      # 14 - anti_collision_bucket - 橙色
        (128, 0, 128),      # 15 - noparking_board - 紫色
    ]
    
    def __init__(self, vis_score_threshold: float = 0.3):
        """
        初始化可视化器
        
        Args:
            vis_score_threshold: 可视化分数阈值
        """
        self.vis_score_threshold = vis_score_threshold
        self.tmp_trackid = []
        self.save_count = 1
        
    def load_data(self, data_path: str) -> Dict:
        """
        加载MV2DFusion数据文件
        
        Args:
            data_path: 数据文件路径，支持.pkl和.json格式
            
        Returns:
            加载的数据字典
        """
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
            
        return data
    
    def load_points_from_file(self, pts_filename: str) -> np.ndarray:
        """
        从文件加载点云数据
        
        Args:
            pts_filename: 点云文件路径
            
        Returns:
            点云数组，形状为(N, 4) [x, y, z, intensity]
        """
        if not os.path.exists(pts_filename):
            print(f"警告: 点云文件不存在: {pts_filename}")
            # 返回空点云
            return np.zeros((0, 4), dtype=np.float32)
            
        # 假设是.bin格式的点云文件
        points = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_images_from_mv2d_format(self, frame_info):
        """
        从MV2DFusion格式加载相机图像
        
        Args:
            frame_info: 帧信息字典
            
        Returns:
            图像列表
        """
        imgs = []
        
        # 相机顺序定义（基于实际数据格式调整）
        cam_order = [
            "CAM_FRONT", "CAM_BACK",  "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW"
        ]
        
        for cam_name in cam_order:
            if cam_name in frame_info.get('cams', {}):
                img_path = frame_info['cams'][cam_name]['data_path']
                # 简化处理：由于可能没有cv2，创建占位图像
                print(f"相机 {cam_name}: {img_path}")
                # 创建占位图像 (高度, 宽度, 通道)
                imgs.append(np.zeros((600, 800, 3), dtype=np.uint8))
            else:
                # 创建空白图像占位
                imgs.append(np.zeros((600, 800, 3), dtype=np.uint8))
        
        return imgs
    
    def extract_lidar2img_from_mv2d(self, frame_info: Dict) -> List[np.ndarray]:
        """
        从MV2DFusion格式提取lidar2img变换矩阵
        
        Args:
            frame_info: 帧信息字典
            
        Returns:
            lidar2img矩阵列表
        """
        lidar2img_list = []
        
        # 相机顺序定义
        cam_order = [
            "CAM_FRONT", "CAM_BACK",  "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW"
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
            else:
                # 创建单位矩阵占位
                lidar2img_list.append(np.eye(4))
        
        return lidar2img_list
    
    def convert_mv2d_pred_format(self, predictions: Dict, frame_idx: int) -> Dict:
        """
        将MV2DFusion预测格式转换为内部格式
        
        Args:
            predictions: MV2DFusion预测结果
            frame_idx: 帧索引
            
        Returns:
            转换后的预测格式
        """
        # 这里需要根据实际的MV2DFusion预测格式进行调整
        # 目前创建一个模拟的格式
        
        # 假设predictions包含每帧的预测结果
        if isinstance(predictions, list) and len(predictions) > frame_idx:
            frame_pred = predictions[frame_idx]
        else:
            frame_pred = predictions
        
        # 转换为内部格式
        def safe_tensor(data):
            if TORCH_AVAILABLE:
                return torch.tensor(data)
            else:
                return TensorLike(data)
        
        pred_box = {
            'img_bbox': {
                'boxes_3d': safe_tensor(frame_pred.get('boxes_3d', [])),
                'scores_3d': safe_tensor(frame_pred.get('scores_3d', [])),
                'labels_3d': safe_tensor(frame_pred.get('labels_3d', [])),
                'cls_scores': safe_tensor(frame_pred.get('cls_scores', [])),
                'instance_ids': safe_tensor([]),  # MV2DFusion没有instance_ids
            },
            'img_bbox_by_img': {
                'boxes_3d': safe_tensor(frame_pred.get('boxes_3d', [])),
                'scores_3d': safe_tensor(frame_pred.get('scores_3d', [])),
            }
        }
        
        return pred_box
    
    def save_pred(self, pts: np.ndarray, imgs_pure: List[np.ndarray], 
                  pred_box: Dict, lidar2img: List[np.ndarray], 
                  gt_bboxes_3d: np.ndarray, gt_labels_3d: np.ndarray, 
                  save_dir: str, every_frame: bool = False):
        """
        保存预测结果可视化
        
        这个函数基于原始的save_pred函数，但适配了MV2DFusion的数据格式
        """
        if every_frame and not hasattr(self, "save_count"):
            self.tmp_trackid = []
            self.save_count = 1
            
        imgs = []
        
        # 过滤低分数的预测结果
        mask_box = pred_box['img_bbox']["scores_3d"] > self.vis_score_threshold
        pred_box['img_bbox']['boxes_3d'] = pred_box['img_bbox']['boxes_3d'][mask_box]
        pred_box['img_bbox']['cls_scores'] = pred_box['img_bbox']["cls_scores"][mask_box]
        pred_box['img_bbox']['labels_3d'] = pred_box['img_bbox']['labels_3d'][mask_box]
        pred_box['img_bbox']['scores_3d'] = pred_box['img_bbox']['scores_3d'][mask_box]
        
        # MV2DFusion没有instance_ids，设置vis_info=None
        vis_info = None
        
        # 设置颜色
        if len(pred_box['img_bbox']['labels_3d']) > 0:
            color = []
            for id in pred_box['img_bbox']['labels_3d'].cpu().numpy().tolist():
                color.append(self.ID_COLOR_MAP[int(id)])
        else:
            color = (255, 0, 0)
        
        pred_bboxes_3d = pred_box['img_bbox']['boxes_3d']
        pred_bboxes_3d_img = pred_box['img_bbox_by_img']['boxes_3d'][
            pred_box['img_bbox_by_img']['scores_3d'] > self.vis_score_threshold
        ]
        
        # 绘制3D框到图像上
        for j, img_origin in enumerate(imgs_pure):
            img = img_origin.copy()
            if len(pred_bboxes_3d) != 0:
                img = draw_lidar_bbox3d_on_img(
                    pred_bboxes_3d,
                    img,
                    lidar2img[j],
                    img_metas=None,
                    color=color,
                    thickness=1,
                )
            if len(pred_bboxes_3d_img) != 0:
                img = draw_lidar_bbox3d_on_img(
                    pred_bboxes_3d_img,
                    img,
                    lidar2img[j],
                    img_metas=None,
                    color=(128, 128, 1),
                    thickness=1,
                )
            imgs.append(img)
        
        # 绘制BEV视图
        bev = draw_lidar_bbox3d_points_on_bev(
            pts,
            pred_bboxes_3d,
            bev_size=imgs[0].shape[0] * 3 * 3,
            color=color,
            bev_range=300,
            track_id=np.array([]),  # 空的track_id
            prev_trackid=self.tmp_trackid,
            do_trans=True,
            pred_bboxes_3d_img=pred_bboxes_3d_img,
            gt_bboxes_3d=gt_bboxes_3d,
            vis_info=vis_info,
        )
        
        # 添加相机名称标签
        cam_names = [
            "front", "back", "front left", "front right", 
            "back left", "back right", "front narrow"
        ]

        for j, name in enumerate(cam_names):
            if j < len(imgs):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)
                
                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
        
        # 拼接图像
        if len(imgs) >= 7:
            empty = np.zeros_like(imgs[6])
            image = np.concatenate(
                [
                    np.concatenate([empty, imgs[6], empty], axis=1),
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
        else:
            # 如果图像数量不足，创建默认布局
            image = np.concatenate(imgs[:min(len(imgs), 6)], axis=1)
        
        h, w, c = image.shape
        image = cv2.resize(image, (w * 3, h * 3))
        image = np.concatenate([bev, image], axis=1)
        
        # 保存图像
        if every_frame:
            cv2.imwrite(f"{save_dir}/{self.save_count}.jpg", image)
            self.save_count += 1
        else:
            cv2.imwrite(save_dir + "/vis.jpg", image)
        
        return 0
    
    def visualize_frame(self, data: Dict, predictions: Dict, frame_idx: int, 
                       save_dir: str, every_frame: bool = False):
        """
        可视化单帧数据
        
        Args:
            data: MV2DFusion数据
            predictions: 预测结果
            frame_idx: 帧索引
            save_dir: 保存目录
            every_frame: 是否按帧保存
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取帧信息
        frame_info = data['infos'][frame_idx]
        
        # 加载点云
        pts = self.load_points_from_file(frame_info['lidar_path_360_ego'])
        
        # 加载图像
        imgs_pure = self.load_images_from_mv2d_format(frame_info)
        
        # 转换预测格式
        pred_box = self.convert_mv2d_pred_format(predictions, frame_idx)
        
        # 提取lidar2img矩阵
        lidar2img = self.extract_lidar2img_from_mv2d(frame_info)
        
        # 提取GT数据
        gt_bboxes_3d = np.array(frame_info['gt_boxes'])
        gt_labels_3d = np.array(frame_info['gt_labels_3d'])
        
        # 调用可视化函数
        self.save_pred(
            pts=pts,
            imgs_pure=imgs_pure,
            pred_box=pred_box,
            lidar2img=lidar2img,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            save_dir=save_dir,
            every_frame=every_frame
        )


def main():
    """
    主函数，示例用法
    """
    # 创建可视化器
    visualizer = MV2DFusionVisualizer(vis_score_threshold=0.3)
    
    # 加载数据
    data_path = "/home/yr/yr/code/cv/object_detection/3D_OD/MV2DFusion-gly/data/Baidu_OD/train_pkl/test_data.json"
    data = visualizer.load_data(data_path)
    
    # 创建模拟预测结果（实际使用时需要替换为真实预测结果）
    predictions = {
        'boxes_3d': [],
        'scores_3d': [],
        'labels_3d': [],
        'cls_scores': []
    }
    
    # 可视化第一帧
    save_dir = "./visualization_output"
    visualizer.visualize_frame(data, predictions, frame_idx=0, save_dir=save_dir)
    
    print(f"可视化结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()



"""
1. 不同数据如果相机顺序不同，需要修改相机顺序
# 相机顺序定义, zicai数据相机顺序
    cam_order = [
        "CAM_FRONT", "CAM_BACK",  "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW"
    ]

代码优化：cam_names和cam_order应该只在类的私有变量中定义一次，其它地方直接调用，不需要多处定义，不便于维护

"""