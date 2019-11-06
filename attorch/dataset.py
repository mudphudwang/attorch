import h5py
import numpy as np
import torch
from collections import defaultdict, namedtuple, Mapping
from glob import glob
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
import os
from .utils import logger, set_seed


class Invertible:
    def inv(self, y):
        raise NotImplemented('Subclasses of Invertible must implement an inv method')


class DataTransform:
    def initialize(self, dataset):
        pass

    def __repr__(self):
        return self.__class__.__name__


class SubsampleNeurons(DataTransform):
    def __init__(self, datakey, idx, axis):
        super().__init__()
        self.idx = idx
        self.datakey = datakey
        self._subsamp = None
        self.axis = axis

    def initialize(self, dataset):
        self._subsamp = []
        for d in dataset.data_keys:
            if d == self.datakey:
                self._subsamp.append([slice(None) for _ in range(self.axis - 1)] + [self.idx, ...])
            else:
                self._subsamp.append(...)

    def __call__(self, item):
        return tuple(it[sub] for sub, it in zip(self._subsamp, item))


class Neurons2Behavior(DataTransform):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __call__(self, item):
        return tuple((item[0], np.hstack((item[1], item[3][~self.idx])), item[2], item[3][self.idx]))


class ToTensor(DataTransform):
    def __call__(self, item):
        return tuple(torch.from_numpy(it) for it in item)


class H5Dataset(Dataset):
    def __init__(self, filename, *data_keys, info_name=None, transform=None):
        self.fid = h5py.File(filename, 'r')
        m = None
        for key in data_keys:
            assert key in self.fid, 'Could not find {} in file'.format(key)
            if m is None:
                m = len(self.fid[key])
            else:
                assert m == len(self.fid[key]), 'Length of datasets do not match'
        self._len = m
        self.data_keys = data_keys

        if info_name is not None:
            self.info = self.fid[info_name]

        if transform is None:
            self.transform = Chain(TransformFromFuncs(), ToTensor())
        else:
            self.transform = transform

        self.transform.initialize(self)

    def __getitem__(self, item):
        return self.transform(tuple(self.fid[d][item] for d in self.data_keys))

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '\n'.join(['Tensor {}: {} '.format(key, self.fid[key].shape)
                          for key in self.data_keys] + ['Transforms: ' + repr(self.transform)])


class TransformDataset(Dataset):

    def transform(self, x, exclude=None):
        for tr in self.transforms:
            if exclude is None or not isinstance(tr, exclude):
                x = tr(x)
        return x

    def invert(self, x, exclude=None):
        for tr in reversed(filter(lambda tr: not isinstance(tr, exclude), self.transforms)):
            if not isinstance(tr, Invertible):
                raise TypeError('Cannot invert', tr.__class__.__name__)
            else:
                x = tr.inv(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '{} m={}:\n\t({})'.format(self.__class__.__name__, len(self), ', '.join(self.data_groups)) \
               + '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']'


class NumpyZSet(TransformDataset):
    def __init__(self, cachedir, *data_groups, transforms=None):
        self.cachedir = cachedir
        tmp = np.load(os.path.join(cachedir, '0.npz'))
        for key in data_groups:
            assert key in tmp, 'Could not find {} in file'.format(key)
        self._len = len(glob('{}/[0-9]*.npz'.format(self.cachedir)))

        self.data_groups = data_groups

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_groups)

    def __getitem__(self, item):
        dat = np.load(os.path.join(self.cachedir, '{}.npz'.format(item)))
        x = self.data_point(*(dat[g] for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __getattr__(self, item):
        dat = np.load(os.path.join(self.cachedir, 'meta.npz'))
        if item in dat:
            item = dat[item]
            if item.dtype.char == 'S':  # convert bytes to univcode
                item = item.astype(str)
            return item
        else:
            raise AttributeError('Item {} not found in {}'.format(item, self.__class__.__name__))


class H5SequenceSet(TransformDataset):
    def __init__(self, filename, *data_groups, transforms=None):
        self._fid = h5py.File(filename, 'r')

        m = None
        for key in data_groups:
            assert key in self._fid, 'Could not find {} in file'.format(key)
            l = len(self._fid[key])
            if m is not None and l != m:
                raise ValueError('groups have different length')
            m = l
        self._len = m

        self.data_groups = data_groups

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_groups)

    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][str(item)]) for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __getattr__(self, item):
        if item in self._fid:
            item = self._fid[item]
            if isinstance(item, h5py._hl.dataset.Dataset):
                item = item[()]
                if item.dtype.char == 'S':  # convert bytes to univcode
                    item = item.astype(str)
                return item
            return item
        else:
            raise AttributeError('Item {} not found in {}'.format(item, self.__class__.__name__))


class DatasetBase(Dataset):

    def __init__(self, input_shape=None, output_shape=None, tier='train', split_seed=0,
                 mode='unlabelled', augment=True):
        assert tier in ['train', 'validation', 'test'], 'tier must be one of ["train", "validation", "test"]'

        self.tier = tier
        self.split_seed = int(split_seed)
        self.augment = augment
        self._default_augment = self.augment

        self.mode = mode
        self._default_mode = self.mode

        self._input_shape = self.standard_input_shape(input_shape)
        self._output_shape = self.standard_output_shape(output_shape)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @staticmethod
    def standard_input_shape(shape):
        return None if shape is None else torch.Size(shape)

    @staticmethod
    def standard_output_shape(shape):
        return None if shape is None else torch.Size(shape)

    @property
    def sample_weights(self):
        return torch.ones(len(self))

    @property
    def mode(self):
        return self._mode

    @property
    def default_mode(self):
        return self._default_mode

    @mode.setter
    def mode(self, mode):
        if mode is None:
            self._mode = self.default_mode
        else:
            assert mode in self.modes, 'mode must be one of {}'.format(self.modes)
            self._mode = mode

    @property
    def modes(self):
        return ['unlabelled']

    @property
    def augment(self):
        return self._augment

    @property
    def default_augment(self):
        return self._default_augment

    @augment.setter
    def augment(self, augment):
        if augment is None:
            self._augment = self.default_augment
        else:
            self._augment = bool(augment)
            logger.info('Setting {} augmentation to {}'.format(self.__class__.__name__, self._augment))


class DynamicDatasetBase(DatasetBase):

    def __init__(self, input_shape=None, output_shape=None, tier='train', split_seed=0,
                 mode='unlabelled', augment=True, max_frames=1, num_frames=None):

        super().__init__(input_shape, output_shape, tier, split_seed, mode, augment)

        self._max_frames = int(max_frames)
        num_frames = self._max_frames if num_frames is None else min(num_frames, self._max_frames)
        self._num_frames = int(num_frames)
        self._default_frames = self._num_frames

    @property
    def max_frames(self):
        return self._max_frames

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def default_frames(self):
        return self._default_frames

    @num_frames.setter
    def num_frames(self, num_frames):
        if num_frames is None:
            self._num_frames = self.default_frames
        else:
            assert num_frames > 0
            self._num_frames = int(min(num_frames, self.max_frames))

    @property
    def frame_index(self):
        if self.num_frames != self.max_frames:
            if self.augment:
                max_start_frame = (self.max_frames - self.num_frames)
                start_frame = np.round(np.random.uniform(0.0, 1.0) * max_start_frame)
                start_frame = start_frame.astype(np.int).item()
            else:
                start_frame = (self.max_frames - self.num_frames) // 2
        else:
            start_frame = 0
        end_frame = start_frame + self.num_frames
        return np.arange(start_frame, end_frame)


def dynamic_dataloader(dataset):

    def loader(batch_size=4, iterations=None, num_workers=1, seed=0, mode=None, num_frames=None,
               augment=None):

        dataset.mode = mode
        dataset.num_frames = num_frames
        dataset.augment = augment

        def worker_init_fn(wid):
            set_seed(num_workers * seed + wid)

        if iterations is None:
            sampler = SequentialSampler(dataset)
        else:
            replacement = False if batch_size * iterations <= len(dataset) else True
            sampler = WeightedRandomSampler(dataset.sample_weights, batch_size * iterations,
                                            replacement=replacement)

        return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, worker_init_fn=worker_init_fn,
                          pin_memory=True)
    return loader
