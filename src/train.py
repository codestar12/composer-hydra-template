from typing import List, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from composer import Trainer, Callback, Logger, ComposerModel
from composer.loggers.logger_destination import LoggerDestination
from composer.core import Algorithm, DataSpec
from composer.core.evaluator import Evaluator
from composer.utils import dist, reproducibility
from pyparsing import Optional
from sympy import OmegaPower


def train(config: DictConfig) -> None:
    """
    Args:
        config (DictConfig): Configuration composed by Hydra

    Returns:
        Optional[float]: Metric score for hyperparameter optimization
    """

    reproducibility.seed_all(config["seed"])

    model: ComposerModel = hydra.utils.instantiate(config.model)

    try:
        # hack to steal vocab size for config
        OmegaConf.register_new_resolver(
            "vocab_size", lambda: model.model.config.vocab_size
        )
    except:
        print("couldn't access vocab size")
        raise

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    train_dataloader = hydra.utils.instantiate(config.dataset.train_dataloader)
    train_dataset = hydra.utils.instantiate(config.dataset.train_dataset)
    train_dataspec: DataSpec = None
    if "train_dataspec" in config.dataset:
        train_dataspec = hydra.utils.instantiate(
            config.dataset.train_dataspec,
            train_dataset.initialize_object(
                config.dataset.train_batch_size // dist.get_world_size(),
                train_dataloader,
            ),
        )
    else:
        train_dataspec = train_dataset.initialize_object(
            config.dataset.train_batch_size // dist.get_world_size(), train_dataloader
        )

    assert not (
        "eval_dataset" in config.dataset and "evaluators" in config.dataset
    ), f"evaluators and eval_dataset found in config.dataset. Use only one \n{OmegaConf.to_yaml(config.dataset)}"

    # Composer can take dataloaders, dataspecs, evaluators, or list of evaluators
    eval_set: Union[DataSpec, List[Evaluator]] = None

    if "eval_dataset" in config.dataset:
        eval_dataloader = hydra.utils.instantiate(config.dataset.eval_dataloader)
        eval_dataset = hydra.utils.instantiate(config.dataset.eval_dataset)
        if "eval_dataspec" in config.dataset:
            eval_set = hydra.utils.instantiate(
                config.dataset.eval_dataspec,
                eval_dataset.initialize_object(
                    config.dataset.eval_batch_size // dist.get_world_size(),
                    eval_dataloader,
                ),
            )
        else:
            eval_set = eval_dataset.initialize_object(
                config.dataset.eval_batch_size // dist.get_world_size(), eval_dataloader
            )
    else:
        evaluators = []
        for _, eval_conf in config.dataset.evaluators.items():
            print(OmegaConf.to_yaml(eval_conf))
            eval_dataloader = hydra.utils.instantiate(config.dataset.eval_dataloader)
            eval_ds = hydra.utils.instantiate(eval_conf.eval_dataset)
            eval_ds = eval_ds.initialize_object(
                config.dataset.eval_batch_size // dist.get_world_size(),
                eval_dataloader,
            )
            evaluator = hydra.utils.instantiate(eval_conf.evaluator, dataloader=eval_ds)
            evaluators.append(evaluator)

        eval_set = evaluators

    logger: List[LoggerDestination] = []
    callbacks: List[Callback] = []
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

    if "callbacks" in config:
        for _, call_conf in config.callbacks.items():
            if "_target_" in call_conf:
                print(f"Instantiating callbacks <{call_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(call_conf))

    scheduler = hydra.utils.instantiate(config.scheduler)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_set,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
    )
    return trainer.fit()
