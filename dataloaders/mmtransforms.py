from mmseg.datasets.pipelines.transforms import Resize,RandomCrop,RandomFlip,Normalize,PhotoMetricDistortion,Pad
import torchvision.transforms as transforms
import torch
import numpy as np
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['img']
        mask = sample['gt_semantic_seg']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'img': img,
                'gt_semantic_seg': mask}


class transform_tr(object):
    def __init__(self,args):
        self.composed_transforms=transforms.Compose([
            Resize(img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            RandomCrop(crop_size=args.crop_size,cat_max_ratio=0.75),
            RandomFlip(prob=0.5),
            PhotoMetricDistortion(),
            Normalize(**img_norm_cfg),
            Pad(size=args.crop_size,pad_val=0,seg_pad_val=255),
            ToTensor()
        ])
    def __call__(self, sample):
        return self.composed_transforms(sample)
class transform_val(object):
    def __init__(self,args):
        self.composed_transforms=transforms.Compose([
            Resize(img_scale=(2048, 1024),keep_ratio=True),
            # RandomFlip(),
            Normalize(**img_norm_cfg),
            ToTensor()
        ])
    def __call__(self, sample):
        return self.composed_transforms(sample)