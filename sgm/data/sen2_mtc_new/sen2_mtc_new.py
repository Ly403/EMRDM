import numpy as np
import tifffile as tiff
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class Sen2_MTC_New_Multi(Dataset):
    def __init__(self, data_root, use_ir=True, mode='train', mono_temporal=-1, multi_temporal=3):
        self.data_root = data_root
        self.mode = mode
        self.use_ir = use_ir
        self.filepair = []
        self.image_name = []
        self.mono_temporal = mono_temporal # 专为消融实验所准备的参数
        assert multi_temporal in [2, 3], "multi_temporal should be 2 or 3"
        self.multi_temporal = multi_temporal
        
        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
                

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        # drop image augmentation
        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        if self.use_ir:
            ret['cond_image'] = torch.stack([image_cloud0, image_cloud1, image_cloud2], dim=0)
        else:
            ret['cond_image'] = torch.stack([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]], dim=0)
        ret['raw_image'] = torch.stack([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]], dim=0)
        ret['path'] = self.image_name[index]+".png"
        # mono_temporal覆盖multi_temporal
        if self.mono_temporal != -1:
            ret['cond_image'] = ret['cond_image'][self.mono_temporal]
            ret['raw_image'] = ret['raw_image'][self.mono_temporal]
        elif self.multi_temporal == 2:
            ret['cond_image'] = ret['cond_image'][:2]
            ret['raw_image'] = ret['raw_image'][:2]
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        # drop image augmentation
        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index//4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image


if __name__ == "__main__":
    data = Sen2_MTC_New_Multi(data_root="/remote-home/share/dmb_nas/liuyi/Sen2_MTC_New/", mode="train")
    print("====================================================")
    for key,val in data[0].items():
        print(key, (val.shape, val.min(), val.max()) if isinstance(val, torch.Tensor) else val)
    print("====================================================")
    data = Sen2_MTC_New_Multi(data_root="/remote-home/share/dmb_nas/liuyi/Sen2_MTC_New/", mode="val")
    for key,val in data[0].items():
        print(key, (val.shape, val.min(), val.max()) if isinstance(val, torch.Tensor) else val)
    print("====================================================")
    data = Sen2_MTC_New_Multi(data_root="/remote-home/share/dmb_nas/liuyi/Sen2_MTC_New/", mode="test")
    for key,val in data[0].items():
        print(key, (val.shape, val.min(), val.max()) if isinstance(val, torch.Tensor) else val)