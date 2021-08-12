from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils.data import TwoGraphData


class BaseModel(LightningModule):
    """
    Base model, only requires the dataset to function
    """

    def __init__(self):
        super().__init__()

    def training_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during training step"""
        return self.shared_step(data)

    def validation_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log("val_" + key, value)
        return ss

    def test_step(self, data: TwoGraphData, data_idx: int) -> dict:
        """What to do during test step"""
        return self.shared_step(data)

    def log_histograms(self):
        """Logs the histograms of all the available parameters"""
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def training_epoch_end(self, outputs):
        """What to do at the end of a training epoch. Logs everything"""
        self.log_histograms()
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("train_epoch_" + i, val, self.current_epoch)

    def validation_epoch_end(self, outputs):
        """What to do at the end of a validation epoch. Logs everything"""
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar("val_epoch_" + i, val, self.current_epoch)

    def test_epoch_end(self, outputs: dict):
        """What to do at the end of a test epoch. Logs everything, saves hyperparameters"""
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            metrics["test_" + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimiser and/or lr schedulers"""
        return AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
        )
