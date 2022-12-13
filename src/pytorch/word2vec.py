import hydra
from transformers import AutoTokenizer

# import wandb


@hydra.main(config_name="config.yaml")
def main(cfg) -> None:
    # wandb.init(project=cfg.ETC.project)
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer)
    print(tokenizer)


if __name__ == "__main__":
    main()
