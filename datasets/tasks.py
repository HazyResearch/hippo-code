import torch
from torch import nn
from torch.nn import functional as F


class Task:
    @staticmethod
    def metrics(outs, y, len_batch=None):
        return {}

    @staticmethod
    def metrics_epoch(outs, y, len_batch=None):
        return {}


class BinaryClassification(Task):
    @staticmethod
    def loss(logits, y, len_batch=None):
        # BCE loss requires squeezing last dimension of logits so it has the same shape as y
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())

    @staticmethod
    def metrics(logits, y, len_batch=None):
        return {'accuracy': torch.eq(logits.squeeze(-1) >= 0, y).float().mean()}

    @staticmethod
    def metrics_epoch(logits, y, len_batch=None):
        return BinaryClassification.metrics(torch.cat(logits), torch.cat(y), len_batch)



class MulticlassClassification(Task):
    @staticmethod
    def loss(logits, y, len_batch=None):
        return F.cross_entropy(logits, y)

    @staticmethod
    def metrics(logits, y, len_batch=None):
        return {'accuracy': torch.eq(torch.argmax(logits, dim=-1), y).float().mean()}

    @staticmethod
    def metrics_epoch(logits, y, len_batch=None):
        return MulticlassClassification.metrics(torch.cat(logits, dim=0), torch.cat(y, dim=0), len_batch)


class MSERegression(Task):
    @staticmethod
    def loss(outs, y, len_batch=None):
        if len_batch is None:
            return F.mse_loss(outs, y)
        else:
            # Computes the loss of the first `lens` items in the batches
            mask = torch.zeros_like(outs, dtype=torch.bool)
            for i, l in enumerate(len_batch):
                mask[i, :l, :] = 1
            outs_masked = torch.masked_select(outs, mask)
            y_masked = torch.masked_select(y, mask)
            return F.mse_loss(outs_masked, y_masked)


