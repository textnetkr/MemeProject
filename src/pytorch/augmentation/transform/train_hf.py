import hydra
import warnings
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from dataloader import load
import wandb

warnings.filterwarnings(action="ignore")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL.name)

    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    args = TrainingArguments(
        **cfg.TRAININGARGS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.PATH.save_dir)

    if cfg.ETC.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    main()
