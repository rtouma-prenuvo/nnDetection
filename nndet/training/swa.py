"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod
from typing import Optional, Union, Callable

from loguru import logger

import torch
from torch.optim.lr_scheduler import _LRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.utilities import rank_zero_warn

from nndet.training.learning_rate import CycleLinear


_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class BaseSWA(StochasticWeightAveraging):
    def __init__(
        self,
        swa_epoch_start: int,
        swa_lrs: float = 1e-4,  # Lightning 2.x requires a positive float
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        update_statistics: Optional[bool] = False,
    ):
        """
        New Base Class for Stochastic Weighted Averaging

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            swa_lrs: Learning rate for SWA (required by Lightning 2.x)
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
                
        Notes: Does not support updating of norm weights after training
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            swa_lrs=swa_lrs,  # Pass the actual learning rate
            annealing_epochs=10,
            annealing_strategy="cos",
            avg_fn=avg_fn,
            device=device,
        )
        self.update_statistics = update_statistics
        logger.info(f"Initialize SWA with swa epoch start {self.swa_start}")

    def pl_module_contains_batch_norm(self, pl_module: 'pl.LightningModule'):
        if self.update_statistics:
            raise NotImplementedError("Updating the statistis of the "
                                      "normalization layer is not suported yet.")
        else:
            return self.update_statistics

    def on_train_epoch_start(self,
                             trainer: 'pl.Trainer',
                             pl_module: 'pl.LightningModule',
                             ):
        """
        Repalce current lr scheduler with SWA scheduler
        """
        if trainer.current_epoch == self.swa_start:
            optimizer = trainer.optimizers[0]
            
            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

            _scheduler = self.get_swa_scheduler(optimizer)
            
            # Store the actual scheduler object for state_dict()
            if isinstance(_scheduler, dict):
                self._swa_scheduler = _scheduler.get("scheduler")
                swa_scheduler_config = _scheduler
            else:
                self._swa_scheduler = _scheduler
                # Default scheduler config for Lightning 2.x
                swa_scheduler_config = {
                    "scheduler": _scheduler,
                    "name": None,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": None,
                    "strict": True,
                }

            # Lightning 2.x: lr_schedulers -> lr_scheduler_configs (now LRSchedulerConfig objects)
            lr_scheduler_configs = getattr(trainer, 'lr_scheduler_configs', getattr(trainer, 'lr_schedulers', []))
            if lr_scheduler_configs:
                # Lightning 2.x uses LRSchedulerConfig objects with .scheduler attribute
                if hasattr(lr_scheduler_configs[0], 'scheduler'):
                    lr_scheduler = lr_scheduler_configs[0].scheduler
                else:
                    lr_scheduler = lr_scheduler_configs[0]["scheduler"]
                rank_zero_warn(f"Swapping lr_scheduler {lr_scheduler} for SWA scheduler")
                # Replace the scheduler in the config object
                if hasattr(lr_scheduler_configs[0], 'scheduler'):
                    lr_scheduler_configs[0].scheduler = self._swa_scheduler
                else:
                    lr_scheduler_configs[0] = swa_scheduler_config
            else:
                # If no schedulers exist, we can skip SWA scheduler swap
                logger.warning("No lr_schedulers found, skipping SWA scheduler swap")

            self.n_averaged = torch.tensor(0, dtype=torch.long, device=pl_module.device)

        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self.avg_fn)

        if trainer.current_epoch == self.swa_end + 1:
            raise NotImplementedError("This should never happen (yet)")

    @abstractmethod
    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        """
        Generate LR scheduler for SWA

        Args:
            optimizer: optimizer to wrap

        Returns:
            Union[_LRScheduler, dict]: If a lr scheduler is returned it will
                be stepped once per epoch. Can also return a whole config of
                the scheduler to customize steps.
        """
        raise NotImplementedError


class SWACycleLinear(BaseSWA):
    def __init__(self,
                 swa_epoch_start: int,
                 cycle_initial_lr: float,
                 cycle_final_lr: float,
                 num_iterations_per_epoch: int,
                 avg_fn: Optional[_AVG_FN] = None,
                 device: Optional[Union[torch.device, str]] = torch.device("cpu"),
                 update_statistics: Optional[bool] = None,
                 ):
        """
        SWA based on :class:`CycleLinear`

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            cycle_initial_lr: initial learning rate of cycle
            cycle_final_lr: final learning rate of cycle
            num_iterations_per_epoch: number of train iterations per epoch
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            swa_lrs=cycle_initial_lr,  # Use cycle_initial_lr for Lightning 2.x
            avg_fn=avg_fn,
            device=device,
            update_statistics=update_statistics,
            )
        self.cycle_initial_lr = cycle_initial_lr
        self.cycle_final_lr = cycle_final_lr
        self.num_iterations_per_epoch = num_iterations_per_epoch

    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        return {
            "scheduler": CycleLinear(
                optimizer=optimizer,
                cycle_num_iterations=self.num_iterations_per_epoch,
                cycle_initial_lr=self.cycle_initial_lr,
                cycle_final_lr=self.cycle_final_lr,
                ),
            "interval": "step",
        }
