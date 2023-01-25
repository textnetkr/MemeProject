import hydra
import numpy as np
import wandb
from dataloader import load
from datasets import load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    args = TrainingArguments(
        **cfg.TRAININGS,
    )

    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.name,
        num_labels=cfg.DATASETS.num_classes,
    )

    # metrics
    def compute_metrics(eval_preds):
        metric = load_metric(cfg.METRICS.metric_name)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return metric.compute(
            predictions=predictions,
            references=labels,
            average=cfg.METRICS.average,
        )

    # train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # model save
    trainer.save_model(cfg.PATH.save_dir)

    if cfg.ETC.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    main()
