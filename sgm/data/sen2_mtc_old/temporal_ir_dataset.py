import os.path
from sgm.data.sen2_mtc_old.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from sgm.data.sen2_mtc_old.image_folder import make_dataset
from PIL import Image
import torch
from argparse import Namespace

class TemporalIrDataset(BaseDataset):
    """A dataset class for temporal image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {{A_0, A_1, A_2, ir},B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w5 = int(w / 5)
        A_0 = AB.crop((0, 0, w5, h))
        A_1 = AB.crop((w5, 0, 2*w5, h))
        A_2 = AB.crop((2*w5, 0, 3*w5, h))
        ir = AB.crop((3*w5, 0, 4*w5, h))
        B = AB.crop((4*w5, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A_0.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A_0 = A_transform(A_0)
        A_1 = A_transform(A_1)
        A_2 = A_transform(A_2)
        ir = A_transform(ir)
        B = B_transform(B)

        # Now split ir into constituent channels
        A_0_ir = ir[0,:,:].unsqueeze(0)
        A_1_ir = ir[1,:,:].unsqueeze(0)
        A_2_ir = ir[2,:,:].unsqueeze(0)
        
        A_raw = torch.stack([A_0,A_1,A_2],dim=0)
        A_0 = torch.cat([A_0,A_0_ir],dim=0)
        A_1 = torch.cat([A_1,A_1_ir],dim=0)
        A_2 = torch.cat([A_2,A_2_ir],dim=0)
        
        A = torch.stack([A_0,A_1,A_2],dim=0)
        
        return {
            'gt_image':B,
            'raw_image':A_raw,
            'cond_image':A,
            'paths': AB_path
        }
        # return {'A_0': A_0, 'A_1': A_1, 'A_2': A_2,'A_0_ir': A_0_ir, 'A_1_ir': A_1_ir, 'A_2_ir': A_2_ir, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class TemporalIrDatasetInterface(TemporalIrDataset):
    def __init__(
        self,
        dataroot,
        input_nc=4,
        output_nc=3,
        direction='AtoB', # AtoB or BtoA
        load_size=286, # scale images to this size
        crop_size=256, # then crop to this size
        max_dataset_size=float('inf'), # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
        preprocess='resize_and_crop',  # scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        no_flip=True, # if specified, do not flip the images for data augmentation
        display_winsize=256, # display window size for both visdom and HTML
        phase='train' # train, val, test, etc
    ):
        opt = Namespace(
            dataroot=dataroot,
            input_nc=input_nc,
            output_nc=output_nc,
            direction=direction,
            load_size=load_size,
            crop_size=crop_size,
            max_dataset_size=max_dataset_size,
            preprocess=preprocess,
            no_flip=no_flip,
            display_winsize=display_winsize,
            phase=phase
        )
        super().__init__(opt)


if __name__ == "__main__":
    data = TemporalIrDatasetInterface(dataroot="/remote-home/share/dmb_nas/liuyi/Sen2_MTC_Old/MultiTemporal")
    for key,val in data[0].items():
        print(key, (val.shape, val.min(), val.max()) if isinstance(val, torch.Tensor) else val)