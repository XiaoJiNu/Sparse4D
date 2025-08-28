# 些脚本是针对训练范围要求，如果标注范围与训练范围要求不同，则将过滤gt，生成对应mask
import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from ..utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)
import cv2
@PIPELINES.register_module()
class VisualizeBEVImageNusences(object):
    def __init__(self, is_vis=True,point2img=True):
        self.is_vis = is_vis
        self.num = 0
        self.point2img = point2img
    def __call__(self, data):
        if not self.is_vis or self.num>100:
            return data
        mmcv.dump([data['points'],data['gt_bboxes_3d']],"./pts_box.pkl")
        pts_all = data['points'][:,:4]
        pts_all[:,3]=1
        if "img" in data:
            for img_id,img in enumerate(data["img"]):
                lidar2img = data['lidar2img'][img_id]
                img_item = np.array(img,dtype=np.uint8)
                gt_box = data['gt_bboxes_3d']
                # gt_box[:,[3,4]] = gt_box[:,[4,3]] 
                img = draw_lidar_bbox3d_on_img(
                        gt_box,
                        img,
                        lidar2img,
                        img_metas=None,
                        color=(255, 0, 0),
                        thickness=3,
                    )
                if self.point2img:
                    # pts_2d = pts_all @ lidar2img.T
                    pts_2d = pts_all @ lidar2img.T

                    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]
                    for pts in pts_2d:
                        if pts[0] > img_item.shape[1] or pts[1]>img_item.shape[0] or pts[0]<0 or pts[1]<0 :
                            continue
                        cv2.circle(img,(int(pts[0]),int(pts[1])),1,(255,0,0),-1)
                cv2.imwrite(f"./show_dirs_test/vis_img1_{img_id}.jpg",img)
        self.num += 1
        return data

@PIPELINES.register_module()
class VisualizeBEVImage(object):
    def __init__(self, is_vis=True):
        self.is_vis = is_vis
        self.num = 0
    def __call__(self, data):
        if not self.is_vis or self.num>100:
            return data
        # colors = self.get_colors()
        dump_data = {}
        pts=data['points']#.tensor.numpy()
        # dump_data['pts_360']=data['points_360'].tensor.numpy()
        box=data['gt_bboxes_3d']#.tensor.numpy()
        mmcv.dump([pts,box],f"./show_dirs_test/dump_data1.pkl")

        # imgs = data['img']
        lidar2img = data['lidar2img']
        gt_boxes_3d_corner =data['gt_bboxes_3d'].corners

        cam_map = [ 'CAM_FRONT','CAM_FRONT_RIGHT',
                        'CAM_FRONT_LEFT','CAM_BACK',
                        'CAM_BACK_LEFT','CAM_BACK_RIGHT', 'CAM_A_FRONT', 'CAM_A_BACK', 'CAM_A_RIGHT','CAM_A_LEFT']

        boxes = data['gt_bboxes_3d']#.tensor #相机坐标系
        labels_ = data['gt_labels_3d']

        box_corner = boxes.corners
        center  = boxes.center
        if "img" in data:
                for img_id,img in enumerate(data["img"]):
                        pts_360 = copy.deepcopy(data['points'].tensor.numpy())

                        img_item = np.array(img,dtype=np.uint8)
                        
                        labels = labels_ 
                        corners = boxes.corners.numpy()
                        num_bboxes = corners.shape[0]
                        pts_360 = pts_360[:,:4]
                        pts_360[:,3] = 1
                        coords = np.concatenate(
                                [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
                        )
                        if 'lidar_aug_matrix' in data:
                                lidar_aug_matrix_item = data['lidar_aug_matrix']#[img_id]
                                lidar_aug_matrix_item_trans = lidar_aug_matrix_item[:3,3]
                                lidar_aug_matrix_item_rot   = lidar_aug_matrix_item[:3,:3]
                                coords[:,:3] = coords[:,:3]-lidar_aug_matrix_item_trans
                                pts_360[:,:3] = pts_360[:,:3]-lidar_aug_matrix_item_trans
                                
                                coords =coords.T
                                pts_360 = pts_360.T
                                pts_360[:3,:] = np.linalg.inv(lidar_aug_matrix_item_rot)@pts_360[:3,:]

                                coords[:3,:] = np.linalg.inv(lidar_aug_matrix_item_rot)@coords[:3,:]
                                coords =coords.T
                                pts_360 = pts_360.T

                        
                        lidar2img_item = lidar2img[img_id]
                        coords = coords @ lidar2img_item.T
                        pts_360 = pts_360 @ lidar2img_item.T

                        coords = coords.reshape(-1, 8, 4)
                        indices = np.all(coords[..., 2] > 0, axis=1)
                        coords = coords[indices]
                        labels = labels[indices]
                        pts_360 = pts_360[pts_360[:,2]>0]

                        indices = np.argsort(-np.min(coords[..., 2], axis=1))
                        coords = coords[indices]
                        labels = labels[indices]

                        coords = coords.reshape(-1, 4)
                        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
                        coords[:, 0] /= coords[:, 2]
                        coords[:, 1] /= coords[:, 2]

                        pts_360[:,2] = np.clip(pts_360[:,2], a_min=1e-5, a_max=1e5)
                        pts_360[:,0] /= pts_360[:,2]
                        pts_360[:,1] /= pts_360[:,2]

                        # coords_bak =copy.deepcopy(coords)

                        if "img_aug_matrix" in data:
                                img_aug_matrix_item = data['img_aug_matrix'][img_id]
                                coords = img_aug_matrix_item@coords.T
                                coords = coords.T
                                coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
                                pts_360 = (img_aug_matrix_item@pts_360.T).T
                                pts_360[:,2] = np.clip(pts_360[:,2], a_min=1e-5, a_max=1e5)

                        coords = coords[..., :2].reshape(-1, 8, 2)

                        for index in range(coords.shape[0]):
                                # name = classes[labels[index]]
                                for start, end in [
                                (0, 1),
                                (0, 3),
                                (0, 4),
                                (1, 2),
                                (1, 5),
                                (3, 2),
                                (3, 7),
                                (4, 5),
                                (4, 7),
                                (2, 6),
                                (5, 6),
                                (6, 7),
                                ]:
                                        cv2.line(
                                                img_item,
                                                coords[index, start].astype(np.int32),
                                                coords[index, end].astype(np.int32),
                                                (255,0,0),
                                                2,
                                                cv2.LINE_AA,
                                        )
                        box2d = data['gt_bboxes'][img_id].astype(np.int32)
                        for box in box2d:
                            cv2.rectangle(img_item, (box[0], box[1]), (box[2], box[3]), (0, 255,0), 2)
                        for pts in pts_360:
                            cv2.circle(img_item,(int(pts[0]),int(pts[1])),1,(255,0,0),-1)
                        # name_key =list(data['cams'].keys())[img_id].replace("CAM_","")
                        cv2.imwrite(f"./show_dirs_test/vis_img1_{img_id}.jpg",img_item)
                        self.num += 1
                # exit()
        return data

@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict

@PIPELINES.register_module()
class GetClassRangeMask(object):
    """根据数据源生成类别掩码和范围掩码的数据处理类
    
    功能：
    针对多数据源训练场景，不同数据源可能只包含部分类别的标注，且检测范围也不同。
    该类根据数据源信息生成对应的类别掩码（class_mask）和范围掩码（range_mask），
    用于在训练时过滤不相关的类别和范围外的目标。
    
    输入参数：
    - classes: list, 所有类别名称列表
    - default_range: list/array, 默认检测范围
    - data_source_with_label: dict, 各数据源包含的类别映射 {数据源名: [类别列表]}
    - data_source_range: dict, 各数据源的检测范围映射 {数据源名: 范围参数}
    
    输出：
    为输入字典添加以下字段：
    - class_mask: np.array, 类别掩码，1表示该类别需要训练，0表示忽略
    - range_mask: np.array, 范围掩码，用于限制目标检测的空间范围
    """
    def __init__(self,classes,default_range,data_source_with_label,data_source_range,):
        self.classes = classes  # 所有类别列表
        self.data_source_with_label = data_source_with_label  # 数据源-类别映射
        self.data_source_range = data_source_range  # 数据源-范围映射
        self.default_range = default_range  # 默认检测范围

        # 构建类别掩码字典，"all"表示使用全部类别
        self.class_mask_all = {"all":[1]*(len(self.classes))}
        # 为每个数据源生成类别掩码
        for datasource,class_item in self.data_source_with_label.items():
            tmp_mask = []
            for clsname in self.classes:
                if clsname in class_item:
                    tmp_mask.append(1)  # 该数据源包含此类别
                else:
                    tmp_mask.append(0)  # 该数据源不包含此类别
            self.class_mask_all[datasource] = tmp_mask

    def __call__(self, input_dict):
        """处理输入数据字典，添加类别掩码和范围掩码
        
        输入：
        - input_dict: dict, 包含训练数据的字典，可能包含"data_source"字段
        
        输出：
        - input_dict: dict, 原字典基础上添加"class_mask"和"range_mask"字段
        """
        # 如果数据字典中包含数据源信息
        if input_dict.get("data_source",False):
            # 检查数据源是否在预定义的掩码字典中
            if input_dict["data_source"] not in self.class_mask_all:
                print(input_dict["data_source"])
                # 未知数据源使用默认设置
                input_dict["class_mask"] = self.class_mask_all["all"]
                input_dict["range_mask"] = self.default_range
            else:
                # 使用对应数据源的类别掩码和范围掩码
                input_dict["class_mask"] = self.class_mask_all[input_dict["data_source"]]
                input_dict["range_mask"] = self.data_source_range[input_dict["data_source"]]

        else:
            # 无数据源信息时使用默认设置
            input_dict["class_mask"] = self.class_mask_all["all"]
            input_dict["range_mask"] = self.default_range
        
        # 转换为numpy数组格式
        input_dict["class_mask"] = np.array(input_dict["class_mask"])
        input_dict["range_mask"] = np.array(input_dict["range_mask"])
        return input_dict

@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init__(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
            # input_dict["focal"] = np.sqrt(
            #     np.abs(np.linalg.det(input_dict["cam_intrinsic"][:, :2, :2]))
            # )
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = DC(
                to_tensor(input_dict["gt_visibility"]).long()
            )
        if "instance_inds_img" in input_dict:
            input_dict["instance_id_img"] = input_dict["instance_inds_img"]
        if "gt_bboxes_3d_img" in input_dict:
            input_dict["gt_bboxes_3d_img"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d_img"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d_img"] = DC(
                to_tensor(input_dict["gt_bboxes_3d_img"]).float()
            )
        if "gt_labels_3d_img" in input_dict:
            input_dict["gt_labels_3d_img"] = DC(
                to_tensor(input_dict["gt_labels_3d_img"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)
        if "points" in input_dict:
            input_dict["points"] = DC(to_tensor(input_dict["points"]))

        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][
                gt_bboxes_mask
            ]
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = input_dict["gt_visibility"][
                gt_bboxes_mask
            ]
        if "gt_labels_3d_img" in input_dict:
            gt_labels_3d_img = input_dict["gt_labels_3d_img"]
            gt_bboxes_img_mask = np.array(
                [n in self.labels for n in gt_labels_3d_img], dtype=np.bool_
            )
            input_dict["gt_bboxes_3d_img"] = input_dict["gt_bboxes_3d_img"][gt_bboxes_img_mask]
            input_dict["gt_labels_3d_img"] = input_dict["gt_labels_3d_img"][gt_bboxes_img_mask]
            if "instance_inds_img" in input_dict:
                input_dict["instance_inds_img"] = input_dict["instance_inds_img"][
                    gt_bboxes_img_mask
                ]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str

@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = input_dict["gt_visibility"][mask]
        
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]

        if "gt_bboxes_3d_img" in input_dict:
            gt_bboxes_3d_img = input_dict["gt_bboxes_3d_img"]
            gt_labels_3d_img = input_dict["gt_labels_3d_img"]
            dist_img = np.sqrt(
                np.sum(gt_bboxes_3d_img[:, :2] ** 2, axis=-1)
            )
            mask_img = np.array([False] * len(dist_img))
            for label_idx, dist_thred in enumerate(self.class_dist_thred):
                mask_img = np.logical_or(
                    mask_img,
                    np.logical_and(gt_labels_3d_img == label_idx, dist_img <= dist_thred),
                )

            gt_bboxes_3d_img = gt_bboxes_3d_img[mask_img]
            gt_labels_3d_img = gt_labels_3d_img[mask_img]

            input_dict["gt_bboxes_3d_img"] = gt_bboxes_3d_img
            input_dict["gt_labels_3d_img"] = gt_labels_3d_img
            if "instance_inds_img" in input_dict:
                input_dict["instance_inds_img"] = input_dict["instance_inds_img"][mask_img]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str

@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str

