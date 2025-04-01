"""
Original work Copyright (c) 2024 littlebeen
Credits to: https://github.com/littlebeen/Cloud-removal-model-collection
Modified by Ly403 at 2025-04-01. 
"""
import numpy as np
import os
from torch.utils import data
from sgm.data.cuhk.imgproc import imresize
import skimage.io as io

class TrainDataset(data.Dataset):

    def __init__(self, datasets_dir, nir_datasets_dir=None, isTrain=True):
        super().__init__()
        if(isTrain):
            self.datasets_dir = datasets_dir+'/train' #change to the path of your dataset
        else:
            self.datasets_dir = datasets_dir+'/test'
        if nir_datasets_dir is not None:
            if(isTrain):
                self.nir_datasets_dir = nir_datasets_dir + '/train'
            else:
                self.nir_datasets_dir = nir_datasets_dir + '/test'

        self.imlistl = sorted(os.listdir(os.path.join(self.datasets_dir, 'label')))
        self.nir_imlistl = sorted(os.listdir(os.path.join(self.nir_datasets_dir, 'label'))) if nir_datasets_dir is not None else None
        assert len(self.imlistl) == len(self.nir_imlistl), 'The number of images in the RGB dataset and NIR dataset should be the same'

    def __getitem__(self, index):
        # a dataset contain 4 bands. it read the nir band and RGB band separately
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlistl[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlistl[index]))).astype(np.float32)
        if self.nir_datasets_dir is not None:
            nirt = io.imread(os.path.join(self.nir_datasets_dir, 'label', str(self.nir_imlistl[index]))).astype(np.float32)[:,:,0]
            nirx = io.imread(os.path.join(self.nir_datasets_dir, 'cloud', str(self.nir_imlistl[index]))).astype(np.float32)[:,:,0]
            t =np.concatenate([t,nirt[:,:,np.newaxis]],axis=2)
            x =np.concatenate([x,nirx[:,:,np.newaxis]],axis=2)
        t = imresize(t, 1/2)
        x = imresize(x, 1/2)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        #M = io.imread(os.path.join(self.datasets_dir, 'mask', str(self.imlistl[index]))).astype(np.float32)
        # M[M>0.5]=1
        # M[M<=0.5]=0
        
        x = (x / 255) * 2 - 1
        t = (t / 255) * 2 - 1
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        cloudy = x[:3,...]
        filename = self.imlistl[index].split('.')[0]

        return {
            "cloudy": cloudy,
            "cond_image": x,
            "label": t,
            "M": M,
            "image_path": filename + ".png"
        }

    def __len__(self):
        return len(self.imlistl)