import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="base", version_base = None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Prepare the dataset
    trainloaders, valloaders = get_data(cfg.data_path_c1, cfg.img_min_c1, cfg.img_max_c1, cfg.train_batch_size, cfg.val_batch_size, cfg.num_workers)
    print(len(trainloaders), len(valloaders))
    # Define the clients

    # Define the strategy

    # Start Simluation

    # Save results
    

if __name__() == "__main__":
    main()
