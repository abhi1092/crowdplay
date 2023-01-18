from d3rlpy.logger import D3RLPyLogger
from typing import Optional

import os
import time
from datetime import datetime
from typing import Dict, Optional

import d3rlpy
import wandb

from d3rlpy.logger import LOG, default_json_encoder

class WandbLogger(d3rlpy.logger.D3RLPyLogger):

    def __init__(
        self,
        experiment_name: str,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        self._save_metrics = save_metrics
        self._verbose = verbose

        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime("%Y%m%d%H%M%S")
                self._experiment_name = experiment_name + "_" + date
            else:
                self._experiment_name = experiment_name

            if self._save_metrics:
                self._logdir = os.path.join(root_dir, self._experiment_name)
                if not os.path.exists(self._logdir):
                    os.makedirs(self._logdir)
                    LOG.info(f"Directory is created at {self._logdir}")
                    break
                if with_timestamp:
                    time.sleep(1.0)
                else:
                    raise ValueError(f"{self._logdir} already exists.")
            else:
                break

        self._metrics_buffer = {}
        self._writer = None
        self._params = None

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print(f"{epoch},{step},{metric}", file=f)

                if self._writer:
                    self._writer.add_scalar(f"metrics/{name}", metric, epoch)

            metrics[name] = metric

        wandb.log({"epoch": epoch, **{f"metrics/{k}": v for k, v in metrics.items()} })

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics


class WandbLoggerWrapper(object):

  def _prepare_logger(
        self,
        save_metrics: bool,
        experiment_name: Optional[str],
        with_timestamp: bool,
        logdir: str,
        verbose: bool,
        tensorboard_dir: Optional[str],
    ) -> D3RLPyLogger:
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = WandbLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            with_timestamp=with_timestamp,
        )

        return logger