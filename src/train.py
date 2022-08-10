from typing import List, Optional

import hydra
from omegaconf import DictConfig
from composer import Trainer, Callback, Logger, ComposerModel
from composer.loggers.logger_destination import LoggerDestination
from composer.core import Algorithm
from pyparsing import Optional


def train(config: DictConfig) -> None:
    """
    Args:
        config (DictConfig): Configuration composed by Hydra

    Returns:
        Optional[float]: Metric score for hyperparameter optimization
    """

    model: ComposerModel = hydra.utils.instantiate(config.model)

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    train_dataloader = hydra.utils.instantiate(config.dataset.train_dataloader)
    train_dataset = hydra.utils.instantiate(config.dataset.train_dataset)
    train_dataspec = hydra.utils.instantiate(
        config.dataset.train_dataspec,
        train_dataset.initialize_object(config.dataset.batch_size, train_dataloader),
    )

    eval_dataloader = hydra.utils.instantiate(config.dataset.eval_dataloader)
    eval_dataset = hydra.utils.instantiate(config.dataset.eval_dataset)
    eval_dataspec = hydra.utils.instantiate(
        config.dataset.eval_dataspec,
        eval_dataset.initialize_object(config.dataset.batch_size, eval_dataloader),
    )

    logger: List[LoggerDestination] = []

    algorithms: List[Algorithm] = []

    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    if "algorithms" in config:
        for _, ag_conf in config.algorithms.items():
            if "_target_" in ag_conf:
                print(f"Instantiating algorithm <{ag_conf._target_}>")
                algorithms.append(hydra.utils.instantiate(ag_conf))
    
    scheduler = hydra.utils.instantiate(config.scheduler)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
    )
    trainer.fit()
