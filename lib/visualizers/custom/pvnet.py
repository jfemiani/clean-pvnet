from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import os
import skimage.color


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
       
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        kpt_2d_gt =  np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        # mask = batch['mask'][0].cpu()
        mask = output['mask'][0].detach().cpu().numpy() > 0

        num, channels, height, width = output['vertex'].shape
        verts = output['vertex'][0].permute(1, 2, 0).reshape(height, width, channels // 2, 2)
        # breakpoint()
        kpi = 10
        vis = 'vec'

        _, ax = plt.subplots(1, num='vis')
        if vis == 'kpt':
            ax.imshow(inp)
            ax.scatter(kpt_2d[:, 0], kpt_2d[:, 1], marker='o', color='b', s=10, facecolor='none')
            ax.scatter(kpt_2d_gt[:, 0], kpt_2d_gt[:, 1], color='g', s=10, edgecolor='none')
        if vis == 'pose':
            ax.imshow(inp)
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        if vis == 'mask':
            ax.imshow(mask, alpha=1, cmap=plt.cm.gray)
        if vis == 'vec':
            v = verts[:, :, kpi, :].cpu().numpy()
            v /= np.linalg.norm(v, axis=2)[..., None]
            angle = np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi) + 0.5
            rgb = skimage.color.hsv2rgb(np.dstack([angle, np.ones_like(angle), np.ones_like(angle)]))
            ax.imshow(rgb * mask[..., None])

        odir = f'{vis}-kp-{kpi}'
        os.makedirs(odir, exist_ok=True)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f'{odir}/{img_id}.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.clf()
        # plt.show(block=False)


    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)





