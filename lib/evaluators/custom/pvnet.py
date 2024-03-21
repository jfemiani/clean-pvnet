from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils import icp_utils
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
from transforms3d.euler import mat2euler


class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']

        model_path = os.path.join(data_root, 'model.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)

        self.diameter = np.loadtxt(os.path.join(data_root, 'diameter.txt')).item()

        self.euler = []
        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None

    def euler_error(self, pose_pred, pose_targets, sym=(1,1,1)):
        # this SHOULD be szxz for proper Euler angles
        # We will use a rotating frame so that symmetry is easier to hanfle
        pose_pred = pose_pred[:3,:3]
        pose_targets = pose_targets[:3,:3]

        # axes = 'rxyx'  - This should be right for symmetry but it is not intuitive to understand
        axes = 'szxz'
        assert sym[0] == 1 and sym[1] == 1, "Right now we only handle Z-symmetry...."
        pred_xyz = np.degrees(mat2euler(pose_pred, axes))
        targ_xyz = np.degrees(mat2euler(pose_targets, axes))
        error_xyz = targ_xyz - pred_xyz

        #   ( [(order*error + 180) % 360-180]) /order
        for i in range(3):
            error_xyz[i] = ((sym[i] * error_xyz[i] + 180) % 360 - 180) / sym[i]

        # Measure the difference
        errors = np.abs(error_xyz)
        self.euler.append(errors)


    def projection_2d(self, pose_pred, pose_targets, K, threshold=cfg.test.projection_threshold):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if self.icp_render is not None:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            self.add_metric(pose_pred_icp, pose_gt, icp=True)
        self.projection_2d(pose_pred, pose_gt, K)
        if cfg.symmetric:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)

        self.euler_error(pose_pred, pose_gt, cfg.symmetry)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        
        euler = np.array(self.euler)
        euler_rmse = np.sqrt(np.mean(euler**2, axis=0))
        euler_one_degree = np.mean(euler<1, axis=0)
        euler_five_degree = np.mean(euler<5, axis=0)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        if self.icp_render is not None:
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
        print("euler:")
        for (i, ax) in enumerate('xyz'):
            print(f" => r{ax}:  mse={euler_rmse[i]:5.2}, 1deg={euler_one_degree[i]:5.2%}, 5deg={euler_five_degree[i]:5.2%}")
       
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_add = []
        self.euler = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap}
