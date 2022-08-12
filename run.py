from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig) -> None:

    from src.train import train 

    return train(config)


if __name__ == "__main__":
    main()
