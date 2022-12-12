import hydra


@hydra.main(config_name="config.yaml")
def main(cfg) -> None:
    input_dim = cfg.MODEL.input_dim
    print(input_dim)


if __name__ == "__main__":
    main()
