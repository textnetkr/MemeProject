import hydra
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from dataloader import load

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # model
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL.name)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    # trainer
    args = TrainingArguments(
        do_train=True,
        do_eval=True if eval_dataset is not None else None,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
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

    trainer.save_model(cfg.PATH.output_dir)


if __name__ == "__main__":
    main()
