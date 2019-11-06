from torch import nn
from .utils import logger


class Net(nn.Module):

    def __init__(self, input_shape=None, output_shape=None):

        super().__init__()

        self._input_shape = self.standard_input_shape(input_shape)
        self._output_shape = self.standard_output_shape(output_shape)
        self._fix = False

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @staticmethod
    def standard_input_shape(shape):
        return shape

    @staticmethod
    def standard_output_shape(shape):
        return shape

    @property
    def params(self):
        raise NotImplementedError

    @property
    def fix(self):
        return self._fix

    @fix.setter
    def fix(self, fix):
        if not isinstance(fix, bool):
            raise ValueError('Expecting <class \'bool\'> but received {}'.format(type(fix)))
        fix_str = 'Fixing' if fix else 'Not fixing'
        logger.info('{} {}'.format(fix_str, self.__class__.__name__))
        self._fix = fix
        self.requires_grad_(not fix)
        self.train(not fix)

    def requires_grad_(self, requires_grad=True):
        if self.fix:
            return super().requires_grad_(False)
        else:
            return super().requires_grad_(requires_grad)

    def train(self, mode=True):
        if self.fix:
            return super().train(False)
        else:
            return super().train(mode)


class NetParams(Net):

    def __init__(self, input_shape=None, output_shape=None, weight_decay=1e-5, dropout=0.0):

        if dropout < 0 or dropout > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout))

        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)

    @property
    def params(self):
        return [{'params': self.parameters(), 'weight_decay': self.weight_decay}]
