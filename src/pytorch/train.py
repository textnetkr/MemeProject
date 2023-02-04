import hydra
import numpy as np
from dataloader import load
import evaluate
import wandb
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    args = TrainingArguments(
        **cfg.TRAININGS,
    )

    # # wandb
    # wandb.init(
    #     project=cfg.ETC.project,
    #     entity=cfg.ETC.entity,
    #     name=cfg.ETC.name,
    # )

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.name,
        num_labels=cfg.MODEL.num_classes,
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

    # trainer
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_dataset["labels"]),
        y=np.array(train_dataset["labels"]),
    )
    print(f"classes : {np.unique(train_dataset['labels'])}")
    print(f"yy : {np.array(train_dataset['labels'])}")
    print(f"class_weights : {len(class_weights)}")

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")

            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(
                    class_weights,
                    dtype=torch.float,
                ).to(device)
            )
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=default_data_collator,
    #     compute_metrics=compute_metrics,
    # )

    trainer.train()

    # model save
    trainer.save_model(cfg.PATH.save_dir)

    if cfg.ETC.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    main()
