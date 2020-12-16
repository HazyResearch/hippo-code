import torch

from omegaconf.dictconfig import DictConfig
from munch import Munch


def remove_postfix(text, postfix):
    if text.endswith(postfix):
        return text[:-len(postfix)]
    return text


# pytorch-lightning returns pytorch 0-dim tensor instead of python scalar
def to_scalar(x):
    return x.item() if isinstance(x, torch.Tensor) else x


def dictconfig_to_munch(d):
    """Convert object of type OmegaConf to Munch so Wandb can log properly
    Support nested dictionary.
    """
    return Munch({k: dictconfig_to_munch(v) if isinstance(v, DictConfig)
                  else v for k, v in d.items()})


def munch_to_dictconfig(m):
    return DictConfig({k: munch_to_dictconfig(v) if isinstance(v, Munch)
                       else v for k, v in m.items()})
