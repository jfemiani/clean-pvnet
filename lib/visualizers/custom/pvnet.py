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

def xsect(p0,n0,p1,n1):
    v1 = p0 - p1
    v2 = n1
    v3 = np.array([-n0[1], n0[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0:
        return p0 + t1 * n0
    return None



class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        # inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        inp = batch['inp'][0].permute(1, 2, 0).cpu().numpy()
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

        kpt_2d_gt = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        # mask = batch['mask'][0].cpu()
        mask = output['mask'][0].detach().cpu().numpy() > 0

        num, channels, height, width = output['vertex'].shape
        verts = output['vertex'][0].permute(1, 2, 0).reshape(height, width, channels // 2, 2)

        kpi = 10
        block = True
        plt.figure(num='vis')
        plt.clf()
        _, ax = plt.subplots(1, num='vis')

        ax.imshow(inp)

        if cfg.visualize.keypoints:
            ax.scatter(kpt_2d[:, 0], kpt_2d[:, 1], marker='o', color='b', s=10, facecolor='none')
            ax.scatter(kpt_2d_gt[:, 0], kpt_2d_gt[:, 1], color='g', s=10, edgecolor='none')
        if cfg.visualize.pose:
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        if cfg.visualize.mask:
            ax.imshow(mask, alpha=cfg.visualize.mask_alpha, cmap=plt.cm.gray)
        if cfg.visualize.angles:
            v = verts[:, :, kpi, :].cpu().numpy()
            v /= np.linalg.norm(v, axis=2)[..., None]
            angle = np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi) + 0.5
            rgb = skimage.color.hsv2rgb(np.dstack([angle, np.ones_like(angle), np.ones_like(angle)]))
            rgba = np.dstack([rgb * mask[..., None], mask[..., None]])
            ax.imshow(rgba)
        if cfg.visualize.vectors:
            v = verts[:, :, kpi, :].cpu().numpy()
            v /= np.linalg.norm(v, axis=2)[..., None]
            angle = np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi) + 0.5
            rgb = skimage.color.hsv2rgb(np.dstack([angle, np.ones_like(angle), np.ones_like(angle)]))
            m = mask>0
            # ax.imshow(rgb * mask[..., None], interpolation='nearest')
            step = cfg.visualize.vectors_spacing
            height, width, _ = inp.shape
            vy, vx = np.mgrid[0:height:step, 0:width:step]
            vu, vv = v[0:height:step, 0:width:step].transpose(2, 0, 1)
            c = rgb[0:height:step, 0:width:step]
            m = m[0:height:step, 0:width:step]
            # ax.scatter(vx, vy, c=rgb.reshape(-1, 3))
            ax.quiver(vx[m], vy[m], vu[m], -vv[m], color=c[m], units='xy', scale=0.5, scale_units='xy', minlength=0, width=0.1)
        if cfg.visualize.points:
            v = verts[:, :, kpi, :].cpu().numpy()
            v /= np.linalg.norm(v, axis=2)[..., None]
            angle = np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi) + 0.5
            rgb = skimage.color.hsv2rgb(np.dstack([angle, np.ones_like(angle), np.ones_like(angle)]))
            m = mask > 0

            height, width, _ = inp.shape
            vy, vx = np.mgrid[0:height, 0:width]
            vy = vy[m]
            vx = vx[m]
            c = rgb[m]
            V = v[m]

            # Some of these points fall way outside of the image bounds
            # So I want to keep the plot from expanding...
            xlim = plt.xlim()
            ylim = plt.ylim()

            for i in range(500):
                i0, i1 = np.random.choice(len(V), 2)
                p0 = np.array([vx[i0], vy[i0]])
                p1 = np.array([vx[i1], vy[i1]])
                v0 = V[i0]
                v1 = V[i1]
                p = xsect(p0, v0, p1, v1)
                if p is not None:
                    # Render marker with low opacity so that it looks vaguely like
                    # a heatmap ...
                    plt.scatter(p[0], p[1], alpha=0.05, lw=0, c='r')
                    if i == 0:
                        plt.plot([p0[0], p[0]], [p0[1], p[1]], c=c[i0], ls='--', alpha=0.5)
                        plt.plot([p1[0], p[0]], [p1[1], p[1]], c=c[i1], ls='--', alpha=0.5)
                        plt.scatter(*p0, c='k', marker='x')
                        plt.scatter(*p1, c='k', marker='x')

            plt.xlim(xlim)
            plt.ylim(ylim)

        plt.axis('off')
        plt.tight_layout(pad=0)

        if cfg.visualize.save:
            odir = cfg.visualize.folder
            os.makedirs(odir, exist_ok=True)
            plt.savefig(os.path.join(odir, f'{img_id}.png'), transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

        if cfg.visualize.show:
            plt.show(block=block)


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





