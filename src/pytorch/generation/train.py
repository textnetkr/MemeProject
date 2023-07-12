import hydra
import torch
from transformers import (
    AutoTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)
    print(f"tokenizer : {tokenizer}")


if __name__ == "__main__":
    main()
