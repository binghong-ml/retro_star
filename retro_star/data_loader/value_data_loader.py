import os
import numpy as np
import torch
import pickle
import logging
from torch.utils.data import Dataset, DataLoader


def unpack_fps(packed_fps):
    # packed_fps = np.array(packed_fps)
    shape = (*(packed_fps.shape[:-1]), -1)
    fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])),
                        axis=-1)
    fps = torch.FloatTensor(fps).view(shape)

    return fps

class ValueDataset(Dataset):
    def __init__(self, fp_value_f):
        assert os.path.exists('%s.pt' % fp_value_f)
        logging.info('Loading value dataset from %s.pt'% fp_value_f)
        data_dict = torch.load('%s.pt' % fp_value_f)
        self.fps = unpack_fps(data_dict['fps'])
        self.values = data_dict['values']

        filter = self.values[:,0] > 0
        self.fps = self.fps[filter]
        self.values = self.values[filter]

        self.reaction_costs = data_dict['reaction_costs']
        self.target_values = data_dict['target_values']
        # self.reactant_fps = unpack_fps(data_dict['reactant_fps'])
        self.reactant_packed_fps = data_dict['reactant_fps']
        self.reactant_masks = data_dict['reactant_masks']
        self.reactant_fps = None
        self.reshuffle()

        assert self.fps.shape[0] == self.values.shape[0]
        logging.info('%d (fp, value) pairs loaded' % self.fps.shape[0])
        logging.info('%d nagative samples loaded' % self.reactant_fps.shape[0])
        print(self.fps.shape, self.values.shape,
              self.reactant_fps.shape, self.reactant_masks.shape)

        logging.info(
            'mean: %f, std:%f, min: %f, max: %f, zeros: %f' %
            (self.values.mean(), self.values.std(), self.values.min(),
             self.values.max(), (self.values==0).sum()*1. / self.fps.shape[0])
        )

    def reshuffle(self):
        shuffle_idx = np.random.permutation(self.reaction_costs.shape[0])
        self.reaction_costs = self.reaction_costs[shuffle_idx]
        self.target_values = self.target_values[shuffle_idx]
        self.reactant_packed_fps = self.reactant_packed_fps[shuffle_idx]
        self.reactant_masks = self.reactant_masks[shuffle_idx]

        self.reactant_fps = unpack_fps(
            self.reactant_packed_fps[:self.fps.shape[0],:,:])

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, index):
        return self.fps[index], self.values[index], \
               self.reaction_costs[index], self.target_values[index], \
               self.reactant_fps[index], self.reactant_masks[index]


class ValueDataLoader(DataLoader):
    def __init__(self, fp_value_f, batch_size, shuffle=True):
        self.dataset = ValueDataset(fp_value_f)

        super(ValueDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def reshuffle(self):
        self.dataset.reshuffle()

