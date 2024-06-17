"""Prepare Custom COCO dataset"""
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from .segbase import SegmentationDataset

class CustomCOCODataset(SegmentationDataset):
    NUM_CLASS = 2  # 根据实际情况更改类别数
    IGNORE_INDEX = -1
    NAME = "coco"

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __init__(self, root='D:/1/chenxiao/ICNet-pytorch-master/coco', split='train', base_size=1024, crop_size=720, mode=None,
                 transform=input_transform):
        super(CustomCOCODataset, self).__init__(root, split, mode, transform, base_size, crop_size)
        assert os.path.exists(self.root), "Error: dataset root path does not exist."
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_{}2017.json'.format(split)))
        self.images = list(self.coco.imgs.keys())

        # 新增：存储图像路径
        self.image_paths = [os.path.join(self.root, '{}2017'.format(split), self.coco.loadImgs(img_id)[0]['file_name'])
                            for img_id in self.images]
        self.mask_paths = [self.generate_mask_path(img_id) for img_id in self.images]

    def generate_mask_path(self, img_id):
        # 根据您的数据集结构生成掩码文件的路径
        # 例如：
        img_info = self.coco.loadImgs(img_id)[0]
        mask_path = os.path.join(self.root, '{}2017_masks'.format(self.split), img_info['file_name'].replace('.jpg', '.png'))
        return mask_path

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.images[index]
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, '{}2017'.format(self.split), img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Load segmentation mask
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask = np.maximum(coco.annToMask(ann) * ann['category_id'], mask)

        mask = Image.fromarray(mask.astype('uint8'))

        # Apply transformations
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        return img, self._mask_transform(mask), img_info['file_name']

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    pass
