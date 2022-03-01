from pycocotools.coco import COCO
from skimage.io import imread
imread('/home/femianjc/vision-server/data/LINEMOD/cat/mask/0000.png')
import numpy as np
im = imread('/home/femianjc/vision-server/data/LINEMOD/cat/mask/0000.png')
np.bincout(im)
np.unique(im.flat)
from PIL import Image
pim = Image.opn('/home/femianjc/vision-server/data/LINEMOD/cat/mask/0000.png')
pim = Image.open('/home/femianjc/vision-server/data/LINEMOD/cat/mask/0000.png')
pim
rim = Image.open('data/custom/mask/000cf99ad7104ebebf067c7f46d1cdcf.png')
rim
np.unique(rim)
np.unique(pim)
ls data/custom/mask/
from glob import glob
masks = glob('data/custom/mask/*.png')
!cp -r data/custom/mask data/custom/mask.old
imread('data/custom/mask/676df46f70d34847ae40098daa72a1b4.png')
imread('data/custom/mask/676df46f70d34847ae40098daa72a1b4.png') > 128
msk = np.uint8(imread('data/custom/mask/676df46f70d34847ae40098daa72a1b4.png') > 128)
msk
imshow(msk)
%pylab
imshow(msk)
np.unique(msk)
!cat /home/femianjc/vision-server/data/LINEMOD/cat/train.json
import jspon
import json
json.load(open('/home/femianjc/vision-server/data/LINEMOD/cat/train.json'))
ds = json.load(open('/home/femianjc/vision-server/data/LINEMOD/cat/train.json'))
ds.keys()
ds['annotations'][0]
def threshold(fn, threshold):
    im = imread(fn)
    fn[fn < threshold] = 0
    imsave(im, fn)
from skimage.io import imread, imsave
fn = masks[0]
fn
threshold(masks[0])
def threshold(fn, level=128):
    im = imread(fn)
    im[im < level] = 0
    im[im > level] = 255
    imsave(im, fn)
threshold(masks[0])
def threshold(fn, level=128):
    im = imread(fn)
    im[im < level] = 0
    im[im > level] = 255
    imsave(fn, im)
threshold(masks[0])
imshow(imread(masks[0])); colorbar()
imshow(imread(masks[1])); colorbar()
imshow(imread(masks[1])); colorbar()
for i, msk in enumerate(masks):
    print("Fixing", i)
    threshold(msk)
imshow(imread(masks[973])); colorbar()
history -f fixed-masks-in-ipython.py
