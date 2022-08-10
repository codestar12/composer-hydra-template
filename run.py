from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig) -> None:

    import src.train as train 

    return train.train(config)


if __name__ == "__main__":
    main()
