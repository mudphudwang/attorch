import h5py
import numpy as np
import torch
from collections import defaultdict, namedtuple, OrderedDict
from glob import glob
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
import os
from .utils import logger, set_seed, all_logging_disabled


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

    def __init__(self, item_shapes=[], augment=False):
        self.item_shapes = item_shapes
        self.augment = bool(augment)

    @property
    def sample_weights(self):
        return torch.ones(len(self))

    @property
    def item_shapes(self):
        return self._item_shapes

    @item_shapes.setter
    def item_shapes(self, item_shapes):
        self._item_shapes = tuple(torch.Size(map(int, s)) for s in item_shapes)

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, augment):
        self._augment = bool(augment)
        logger.info('Setting {} augmentation to {}'.format(self.__class__.__name__, self._augment))


class DynamicDatasetBase(DatasetBase):

    def __init__(self, item_shapes=[], augment=False, num_frames=None):
        super().__init__(item_shapes, augment)
        self.num_frames = num_frames

    @property
    def num_frames(self):
        return self._num_frames

    @num_frames.setter
    def num_frames(self, num_frames):
        if num_frames is None:
            self._num_frames = None
        else:
            assert num_frames > 0
            self._num_frames = int(num_frames)
        logger.info('Setting frame size to {}'.format(self._num_frames))


def dynamic_dataloader(dataset, batch_size=4, iterations=None, num_workers=1, num_frames=None, augment=None):

    if num_frames is not None:
        dataset.num_frames = num_frames

    if augment is not None:
        dataset.augment = augment

    def loader(seed=0):

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
