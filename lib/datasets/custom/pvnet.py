import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image, ImageEnhance
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
from lib.config import cfg


class Dataset(data.Dataset):

    def __init__(self, data_root, ann_file, split, transforms, downsample=0, **kwargs):
        # See lib/datasets/dataset_catalog.py to see where the arguments are specified
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.downsample = downsample
        self.cfg = cfg
        print(f"Loaded Dataset of {len(self.img_ids)} images from {ann_file}")

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']

        # Path should be relative to self.data_root
        path = os.path.normpath(os.path.join(self.data_root, path))

        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = 1  # linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask_path = os.path.join(self.data_root, anno['mask_path'])
        mask = pvnet_data_utils.read_linemod_mask(mask_path, anno['type'], cls_idx)

        if self.downsample > 0:
            # Simulate a much smaller target that was then scaled back up....
            inp = inp.resize((inp.width//self.downsample, inp.height//self.downsample)).resize(inp.size, resample=Image.BICUBIC)

        if cfg.test.change_contrast != 1:
            inp = ImageEnhance.Contrast(inp).enhance(cfg.test.change_contrast)

        return inp, kpt_2d, mask

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask = self.read_data(img_id)
        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img


        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((len(kpt_2d), 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask
