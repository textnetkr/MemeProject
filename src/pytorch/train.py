import hydra
import numpy as np
from dataloader import load
import evaluate
import wandb
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
    precision_metric = evaluate.load(cfg.METRICS.metric_name)

    def compute_metrics(eval_preds):
        logits, labels = eval_preds

        pred = np.argmax(logits, axis=-1)
        print(f"pred : {pred}")
        print(f"labels : {labels}")

        results = precision_metric.compute(
            references=pred,
            predictions=labels,
            average=cfg.METRICS.average,
        )
        return results

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
