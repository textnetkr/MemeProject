import hydra
import pandas as pd
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from dataloader import TextStyleTransferDataset
from sklearn.model_selection import train_test_split


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # data load
    df = pd.read_csv(cfg.PATH.dataset, sep="\t")

    # 값 두 개 미만 제외
    row_notna_count = df.notna().sum(axis=1)
    row_notna_count.plot.hist(bins=row_notna_count.max())
    df = df[row_notna_count >= 2]

    # train, test set
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    print(len(df_train), len(df_test))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # dataloader
    train_dataset = TextStyleTransferDataset(
        df_train,
        tokenizer,
        cfg.DATASET.max_length,
    )
    test_dataset = TextStyleTransferDataset(
        df_test,
        tokenizer,
        cfg.DATASET.max_length,
    )

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL.name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # trainer
    print(f"TRAININGARGS : {cfg.TRAININGARGS}")
    training_args = Seq2SeqTrainingArguments(**cfg.TRAININGARGS)
    # wandb
    wandb.init(
        project=cfg.ETC.project,
        entity=cfg.ETC.entity,
        name=cfg.ETC.name,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model()

    if cfg.ETC.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    main()
