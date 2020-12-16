from pathlib import Path
project_root = Path(__file__).parent.absolute()
import os
# Add to $PYTHONPATH so that ray workers can see
os.environ['PYTHONPATH'] = str(project_root) + ":" + os.environ.get('PYTHONPATH', '')

import numpy as np
import torch
import pytorch_lightning as pl

import hydra
from omegaconf import OmegaConf

from model.model import Model
from datasets import DatasetBase
from model.exprnn.parametrization import get_parameters
from utils import to_scalar


class RNNTraining(pl.LightningModule):

    def __init__(self, model_args, dataset_cfg, train_args):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cfg = dataset_cfg
        self.dataset = DatasetBase.registry[dataset_cfg.name](dataset_cfg)
        self.train_args = train_args
        self.model_args = model_args
        # self.model_args.cell_args.max_length = self.dataset.N # TODO fix datasets
        # cell_args = model_args.cell_args
        # other_args = {k: v for k, v in model_args.items() if k not in ['cell', 'cell_args', 'dropout']}
        self.model = Model(
            self.dataset.input_size,
            self.dataset.output_size,
            # model_args.cell,
            # cell_args=cell_args,
            output_len=self.dataset.output_len,
            # dropout=model_args.dropout,
            # max_length=self.dataset.N,
            **model_args,
        )

    def forward(self, input):
        self.model.forward(input)

    def _shared_step(self, batch, batch_idx, prefix='train'):
        batch_x, batch_y, *len_batch = batch
        # Either fixed length sequence or variable length
        len_batch = None if not len_batch else len_batch[0]
        out = self.model(batch_x, len_batch)
        loss = self.dataset.loss(out, batch_y, len_batch)
        metrics = self.dataset.metrics(out, batch_y)
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return (self._shared_step(batch, batch_idx, prefix='val') if dataloader_idx == 0 else
                self._shared_step(batch, batch_idx, prefix='test'))

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix='test')

    def configure_optimizers(self):
        name_to_opt = {'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
        optimizer = name_to_opt[self.train_args.optimizer]
        if self.model_args.cell == 'exprnn' or self.model_args.cell_args.get('orthogonal', False):
            non_orth_params, log_orth_params = get_parameters(self.model)
            return optimizer([
                {'params': non_orth_params, 'lr': self.train_args.lr, 'weight_decay': self.train_args.wd},
                # {'params': log_orth_params, 'lr': self.train_args.lr_orth},
                {'params': log_orth_params, 'lr': self.train_args.lr/10.0},
            ])
        else:
            return optimizer(self.model.parameters(), lr=self.train_args.lr)

    def prepare_data(self):
        self.dataset.prepare_data()
        kwargs = {'num_workers': self.dataset_cfg.num_workers, 'pin_memory': True}
        self.dataset.prepare_dataloader(self.train_args.batch_size, **kwargs)

    def train_dataloader(self):
        return self.dataset.train_loader

    def val_dataloader(self):
        return [self.dataset.val_loader, self.dataset.test_loader]

    def test_dataloader(self):
        return self.dataset.test_loader


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    if cfg.runner.name == 'pl':
        from pl_runner import pl_train
        trainer, model = pl_train(cfg, RNNTraining)
    elif cfg.runner.name == 'ray':
        # Shouldn't need to install ray unless doing distributed training
        from ray_runner import ray_train
        ray_train(cfg, RNNTraining)
    else:
        assert False, 'Only pl and ray runners are supported'


if __name__ == "__main__":
    main()
