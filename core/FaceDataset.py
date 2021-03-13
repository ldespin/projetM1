import h5py as h5
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict


class FaceDataset(Dataset):
    def __init__(self,data_path,data_type,modality):
        f = h5.File(data_path,'r')
        self.data = np.copy(np.asarray(f[modality]).transpose(0, 3, 1, 2))

        self.gt = np.copy(np.asarray(f['noOcclusion']).transpose(0, 3, 1, 2))
        self.true_label = np.copy(f['labels'])

        self.labels = None
        if data_type == 'val' or data_type == 'test':
            self.labels = np.copy(f['labels'])

        f.close()

    def __getitem__(self, index):

        sample = OrderedDict()

        sample['input'] = self.data[index]

        sample['output'] = self.gt[index]
 
        if self.labels is not None:
            sample['label'] = self.labels[index]
        else:
            sample['label'] = []

        return sample

    def __len__(self):
        return len(self.data)