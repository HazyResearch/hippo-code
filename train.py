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
        self.model = Model(
            self.dataset.input_size,
            self.dataset.output_size,
            output_len=self.dataset.output_len,
            **model_args,
        )

    def forward(self, input):
        self.model.forward(input)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, *len_batch = batch
        # Either fixed length sequence or variable length
        len_batch = None if not len_batch else len_batch[0]
        out = self.model(batch_x, len_batch)
        loss = self.dataset.loss(out, batch_y, len_batch)
        metrics = self.dataset.metrics(out, batch_y)
        return {'loss': loss, 'size': batch_x.shape[0], 'out': out, 'target': batch_y,
                'progress_bar': metrics, 'log': metrics}

    def training_epoch_end(self, outputs, prefix='train'):
        losses = torch.stack([output['loss'] for output in outputs])
        sizes = torch.tensor([output['size'] for output in outputs], device=losses.device)
        loss_mean = (losses * sizes).sum() / sizes.sum()
        outs = [output['out'] for output in outputs]
        targets = [output['target'] for output in outputs]
        metrics = self.dataset.metrics_epoch(outs, targets)
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        results = {f'{prefix}_loss': loss_mean, **metrics}
        results_scalar = {k: to_scalar(v) for k, v in results.items()}  # PL prefers torch.Tensor while we prefer float
        setattr(self, f'_{prefix}_results', results_scalar)
        if getattr(self.train_args, 'verbose', False):
            print(f'{prefix} set results:', results_scalar)
        return {f'{prefix}_loss': loss_mean, 'log': results}

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, *len_batch = batch
        # Either fixed length sequence or variable length
        len_batch = None if not len_batch else len_batch[0]
        out = self.model(batch_x, len_batch)
        loss = self.dataset.loss(out, batch_y, len_batch)
        metrics = self.dataset.metrics(out, batch_y)
        return {'size': batch_x.shape[0], 'loss': loss, 'out': out, 'target': batch_y, **metrics}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs, prefix='val')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs, prefix='test')

    def configure_optimizers(self):
        name_to_opt = {'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
        optimizer = name_to_opt[self.train_args.optimizer]
        if self.model_args.cell == 'exprnn' or self.model_args.cell_args.get('orthogonal', False):
            non_orth_params, log_orth_params = get_parameters(self.model)
            return optimizer([
                {'params': non_orth_params, 'lr': self.train_args.lr, 'weight_decay': self.train_args.wd},
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
        return self.dataset.val_loader

    def test_dataloader(self):
        return self.dataset.test_loader


@hydra.main(config_path="cfg/config.yaml", strict=False)
def main(cfg: OmegaConf):
    print(cfg.pretty())
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
