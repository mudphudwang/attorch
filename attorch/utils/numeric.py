import random
import numpy as np
import torch

from .logging import logger


def set_seed(seed, cuda=True):
    logger.info('Setting numpy and torch seed to {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed(int(seed))
