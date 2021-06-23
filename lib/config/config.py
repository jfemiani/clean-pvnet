from yacs.config import CfgNode as CN
import argparse
import os
import open3d

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'
cfg.det_model = ''
cfg.kpt_model = ''

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()
cfg.heads.vote_dim = 18
cfg.heads.seg_dim = 2
cfg.heads.ct_hm = 30
cfg.heads.wh = 2

# task
cfg.task = ''

# gpus
cfg.gpus = [0, 1, 2, 3]

# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 5
cfg.eval_ep = 5

cfg.demo_path = 'demo_images/cat'

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.milestones = [80, 120]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

#augmentation
cfg.train.affine_rate = 0.
cfg.train.cropresize_rate = 0.
cfg.train.rotate_rate = 0.
cfg.train.rotate_min = -30
cfg.train.rotate_max = 30

cfg.train.overlap_ratio = 0.8
cfg.train.resize_ratio_min = 0.8
cfg.train.resize_ratio_max = 1.2

cfg.train.batch_sampler = ''

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.icp = False
cfg.test.un_pnp = False
cfg.test.vsd = False
cfg.test.det_gt = False
cfg.test.projection_threshold = 5
cfg.test.change_contrast = 1.0

cfg.test.batch_sampler = ''

cfg.det_meta = CN()
cfg.det_meta.arch = 'dla'
cfg.det_meta.num_layers = 34
cfg.det_meta.heads = CN({'ct_hm': 1, 'wh': 2})

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

# dataset
cfg.cls_type = 'glue'   # 'glue' and 'eggbox' are symmtric -- 'cat' is not. \
cfg.symmetric = False
cfg.symmetry = (1,1,1)

cfg.visualize = CN()
cfg.visualize.mask = False  # Render the mask over the input
cfg.visualize.mask_alpha = 1  # Opacity of the mask
cfg.visualize.keypoints = False  # Show gt and target keypoints
cfg.visualize.pose = True  # Show bounding boxes
cfg.visualize.vectors = False  # Show quivers for the vectors
cfg.visualize.vectors_spacing = 3  # Integer gap between quivers for the vectors
cfg.visualize.angles = False  # show angles as hue
cfg.visualize.points = False  # show many samples of ray-intersection (vote) points
cfg.visualize.show = True  # Show the plot for each one
cfg.visualize.save = True
cfg.visualize.folder = './vis'  # Save plots to this folder

cfg.custom = CN()
cfg.custom.test = CN()
cfg.custom.test.data_root = 'data/custom'
cfg.custom.test.ann_file = 'data/custom/val.json'

cfg.custom.train = CN()
cfg.custom.train.data_root = 'data/custom'
cfg.custom.train.ann_file = 'data/custom/train.json'

# tless
cfg.tless = CN()
cfg.tless.pvnet_input_scale = (256, 256)
cfg.tless.scale_train_ratio = (1.8, 2.4)
cfg.tless.scale_ratio = 2.4
cfg.tless.box_train_ratio = (1.0, 1.2)
cfg.tless.box_ratio = 1.2
cfg.tless.rot = 360.
cfg.tless.ratio = 0.8

_heads_factory = {
    'pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct_pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct': CN({'ct_hm': 30, 'wh': 2})
}


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if cfg.task in _heads_factory:
        default_heads = _heads_factory[cfg.task]
        if isinstance(cfg.heads, CN):
            default_heads.update(cfg.heads)
        cfg.heads = default_heads

    if 'Tless' in cfg.test.dataset and cfg.task == 'pvnet':
        cfg.cls_type = '{:02}'.format(int(cfg.cls_type))

    if 'Ycb' in cfg.test.dataset and cfg.task == 'pvnet':
        cfg.cls_type = '{}'.format(int(cfg.cls_type))

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    opts_idx = [i for i in range(0, len(args.opts), 2) if args.opts[i].split('.')[0] in cfg.keys()]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    cfg.merge_from_list(opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
