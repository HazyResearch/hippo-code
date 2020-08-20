import os
dir_path = os.path.dirname(os.path.abspath(__file__))

import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from . import copying, adding
from . import utils
from .tasks import BinaryClassification, MulticlassClassification, MSERegression


class DatasetBase():
    registry = {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes with @name attribute
        if hasattr(cls, 'name'):
            cls.registry[cls.name] = cls

    def __init__(self, dataset_cfg, path=dir_path):
        self.dataset_cfg = dataset_cfg
        self.path = path

    def prepare_data(self):
        raise NotImplementedError

    def split_train_val(self, ratio=0.9):
        train_len = int(len(self.train) * ratio)
        self.train, self.val = torch.utils.data.random_split(self.train, (train_len, len(self.train) - train_len))

    def prepare_dataloader(self, batch_size, **kwargs):
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val, batch_size=batch_size, shuffle=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=False, **kwargs)

    def __str__(self):
        return self.name if hasattr(self, 'name') else self.__name__


class MNIST(DatasetBase, MulticlassClassification):
    name = 'mnist'
    input_size = 1
    output_size = 10
    output_len = 0
    N = 784

    def prepare_data(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Lambda(lambda x: x.view(self.input_size, self.N).t())]  # (N, input_size)
        if self.dataset_cfg.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = utils.bitreversal_permutation(self.N)
            transform_list.append(transforms.Lambda(lambda x: x[permutation]))
        transform = transforms.Compose(transform_list)
        self.train = datasets.MNIST(f'{self.path}/{self.name}', train=True, download=True, transform=transform)
        self.test = datasets.MNIST(f'{self.path}/{self.name}', train=False, transform=transform)
        self.split_train_val()

    def __str__(self):
        return f"{'p' if self.dataset_cfg.permute else 's'}{self.name}"


class Copying(DatasetBase, MulticlassClassification):
    name = 'copying'

    def __init__(self, dataset_cfg, path=dir_path):
        super().__init__(dataset_cfg, path)
        self.input_size = dataset_cfg.A
        self.output_size = dataset_cfg.A
        self.output_len = dataset_cfg.M
        self.N = dataset_cfg.L + 2 * dataset_cfg.M

    def prepare_data(self):
        cfg = self.dataset_cfg
        self.train = copying.copying_static_dataset(cfg.L, cfg.M, cfg.A, cfg.variable, cfg.samples)
        self.test = copying.copying_static_dataset(cfg.L, cfg.M, cfg.A, cfg.variable, cfg.test_samples)
        self.split_train_val()

    def __str__(self):
        return f"{self.name}{self.dataset_cfg.L}{'v' if self.dataset_cfg.variable else ''}"


class Adding(DatasetBase, MSERegression):
    name = 'adding'

    def __init__(self, dataset_cfg, path=dir_path):
        super().__init__(dataset_cfg, path)
        self.input_size = 2
        self.output_size = 1
        self.output_len = 0
        self.N = dataset_cfg.L

    def prepare_data(self):
        cfg = self.dataset_cfg
        self.train = adding.adding_static_dataset(cfg.L, cfg.samples)
        self.test = adding.adding_static_dataset(cfg.L, cfg.test_samples)
        self.split_train_val()

    def __str__(self):
        return f"{self.name}{self.dataset_cfg.L}"


# Wrap the data loader with callback function
class LoaderWCallback:
    def __init__(self, loader, callback_fn):
        self.loader = loader
        self.callback_fn = callback_fn

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        return self

    def __next__(self):
        return self.callback_fn(next(self.loader_iter))


class IMDB(DatasetBase, BinaryClassification):
    name = 'imdb'
    output_size = 1
    output_len = 0

    def __init__(self, dataset_cfg, path=dir_path):
        super().__init__(dataset_cfg, path)
        self.input_size = dataset_cfg.vocab_size
        self.N = dataset_cfg.max_length

    # https://github.com/bentrevett/pytorch-sentiment-analysis/issues/6
    def tokenize_once(self):
        import torchtext
        from torchtext import data
        TEXT = data.Field(tokenize='spacy')
        LABEL = data.LabelField()
        train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL, root=f'{self.path}')
        train_examples = [vars(t) for t in train_data]
        test_examples = [vars(t) for t in test_data]
        import json
        with open(f'{self.path}/{self.name}/train.json', 'w+') as f:
            for example in train_examples:
                json.dump(example, f)
                f.write('\n')
        with open(f'{self.path}/{self.name}/test.json', 'w+') as f:
            for example in test_examples:
                json.dump(example, f)
                f.write('\n')

    def prepare_data(self):
        if not os.path.exists(f'{self.path}/{self.name}/train.json'):
            self.tokenize_once()
        import torchtext
        from torchtext import data
        TEXT = data.Field(batch_first=True, include_lengths=True)
        LABEL = data.LabelField(dtype=torch.float)
        fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
        self.train, self.test = data.TabularDataset.splits(
            path = f'{self.path}/{self.name}',
            train = 'train.json',
            test = 'test.json',
            format = 'json',
            fields = fields
        )
        self.train, self.val = self.train.split(0.9)
        TEXT.build_vocab(self.train, max_size=self.input_size - 2)  # Need 2 extra for <unk> and <pad>
        LABEL.build_vocab(self.train)

    def prepare_dataloader(self, batch_size, **kwargs):
        from torchtext import data
        self.train_loader, self.val_loader, self.test_loader = data.BucketIterator.splits(
            (self.train, self.val, self.test),
            shuffle=True,
            sort_key=lambda ex: len(ex.text),
            batch_size = batch_size)

        def postprocess(batch):  # make the loader from torchtext compatible with Pytorch's loader
            x, lens = batch.text
            x = x[:self.N]
            lens = torch.clamp(lens, max=self.N)
            return x, batch.label, lens

        self.train_loader = LoaderWCallback(self.train_loader, postprocess)
        self.val_loader = LoaderWCallback(self.val_loader, postprocess)
        self.test_loader = LoaderWCallback(self.test_loader, postprocess)


class CharacterTrajectories(DatasetBase, MulticlassClassification):
    """ CharacterTrajectories dataset from the UCI Machine Learning archive.

    See datasets.uea.postprocess_data for dataset configuration settings.
    """
    name = 'ct'
    input_size  = 3
    output_size = 20
    output_len = 0


    def __init__(self, dataset_cfg, path=dir_path):
        super().__init__(dataset_cfg, path)
        if self.dataset_cfg.timestamp:
            self.input_size += 1

    def prepare_data(self):
        from datasets import uea

        cfg = self.dataset_cfg
        *data, num_classes, input_channels = uea.get_data(
            'CharacterTrajectories',
            intensity=False,
        )
        train_dataset, val_dataset, test_dataset = uea.postprocess_data(
            *data,
            train_hz=cfg.train_hz,
            eval_hz=cfg.eval_hz,
            train_uniform=cfg.train_uniform,
            eval_uniform=cfg.eval_uniform,
            timestamp=cfg.timestamp,
            train_ts=cfg.train_ts,
            eval_ts=cfg.eval_ts,
        )
        self.train = train_dataset
        self.val   = val_dataset
        self.test  = test_dataset
        assert num_classes == self.output_size, f"Output size should be {num_classes}"
