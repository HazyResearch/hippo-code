import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from model.memory import LTICell, LSICell
from model.op import transition


class OPLTICell(LTICell):
    # name = 'lagt'
    measure = None

    def __init__(self, input_size, hidden_size, memory_size=1, memory_order=-1, measure_args={},
                 **kwargs
                 ):
        if memory_order < 0:
            memory_order = hidden_size

        # A, B = transition(type(self).measure, memory_order)
        A, B = transition(type(self).measure, memory_order, **measure_args)
        super().__init__(input_size, hidden_size, memory_size, memory_order, A, B, **kwargs)
class OPLSICell(LSICell):
    # name = 'lagt'
    measure = None

    def __init__(self, input_size, hidden_size, memory_size=1, memory_order=-1, measure_args={},
                 **kwargs
                 ):
        if memory_order < 0:
            memory_order = hidden_size

        A, B = transition(type(self).measure, memory_order, **measure_args)
        super().__init__(input_size, hidden_size, memory_size, memory_order, A, B, **kwargs)

# TODO there should be a way to declare the parent class programatically to avoid duplicating this
# i.e. have a single OPCell that calls the appropriate superclass constructor
# for measure in ['lagt', 'legt', 'legs']:
#     type('t'+measure, OPLTICell, {'measure': measure}):
#     type('s'+measure, OPLSICell, {'measure': measure}):

class LegendreTranslateCell(OPLTICell):
    name = 'legt'
    measure = 'legt'
class LegendreTranslateSCell(OPLSICell):
    name = 'legts'
    measure = 'legt'
class LegendreScaleCell(OPLSICell):
    name = 'legs'
    measure = 'legs'
class LegendreScaleTCell(OPLTICell):
    name = 'legst'
    measure = 'legs'
class LaguerreTranslateCell(OPLTICell):
    name = 'lagt'
    measure = 'lagt'
class LaguerreTranslateSCell(OPLSICell):
    name = 'lagts'
    measure = 'lagt'
class LMUTCell(OPLTICell):
    name = 'lmut'
    measure = 'lmu'
class LMUCell(OPLTICell):
    name = 'lmu'
    measure = 'lmu'

    def default_initializers(self):
        return {
            'uxh': 'uniform',
            'ux': 'one',
            'uh': 'zero',
            'um': 'zero',
            'hxm': 'xavier',
            'hx': 'zero',
            'hh': 'zero',
            'hm': 'xavier',
        }

    def default_architecture(self):
        return {
            'ux': True,
            'um': True,
            'hx': True,
            'hm': True,
            'hh': True,
            'bias': False,
        }

    def __init__(self, input_size, hidden_size, theta=100, dt=1., **kwargs):
        super().__init__(input_size, hidden_size, dt=dt/theta, **kwargs)


class LegendreScaleNoiseCell(LTICell):
    name = 'legsn'
    measure = 'legs'

    def __init__(self, input_size, hidden_size, memory_size=1, memory_order=-1,
                 **kwargs
                 ):
        if memory_order < 0:
            memory_order = hidden_size

        A, B = transition(type(self).measure, memory_order)
        N = memory_order
        A = A + np.random.normal(size=(N, N)) / N

        super().__init__(input_size, hidden_size, memory_size, memory_order, A, B, **kwargs)
