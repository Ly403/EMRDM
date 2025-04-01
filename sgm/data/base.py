from functools import partial
from sgm.util import instantiate_from_config
import torch
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader, Subset
import pytorch_lightning as pl

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    seed = np.random.get_state()[1][0] + worker_id
    random.seed(int(seed))
    return np.random.seed(int(seed))
    
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=True,
                 shuffle_val_dataloader=False, max_samples_count=1e9, max_samples_frac=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.f_generator = None # For train
        self.g_generator = None # For val and test
        self.max_samples_count = max_samples_count
        self.max_samples_frac = max_samples_frac

    def set_generator(self, f, g=None):
        self.f_generator = f
        self.g_generator = g if g is not None else f

    def prepare_data(self):
        pass
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict()
        for k in self.dataset_configs:
            self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
            self.datasets[k] = Subset(self.datasets[k], range(0, min(len(self.datasets[k]), self.max_samples_count, self.max_samples_frac * len(self.datasets[k]))))
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          generator=self.f_generator,
                          worker_init_fn=init_fn, pin_memory=True)

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          generator=self.g_generator,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          pin_memory=True)

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,generator=self.g_generator,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, pin_memory=True)

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,generator=self.g_generator,
                          num_workers=self.num_workers, worker_init_fn=init_fn, pin_memory=True)
