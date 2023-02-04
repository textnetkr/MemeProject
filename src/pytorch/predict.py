import hydra
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    print("load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)
    print("tokenizer loading done!")

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.PATH.save_dir,
        num_labels=cfg.DATASETS.num_classes,
    )
    model.eval().cuda()
    print("model loading done!")

    # predict
    sentence = "오늘 날씨가 너무 좋은데??"

    data = tokenizer(
        sentence,
        max_length=cfg.DATASETS.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        data = {k: v.cuda() for k, v in data.items()}
        outputs = model(**data)

        print(f"predict : {outputs.logits[0]}")
        # predict = np.argmax(outputs.logits[0].cpu().numpy())
        # print(f"predict : {predict}")

        # meme extract
        # df = fm.load(cfg.PATH.origin_class_ref)
        # df_ref = df[df.g_num.values == outputs]
        # temp_ref = df_ref[df_ref.u.values == i[1]['u']]
        # print(df_ref["u"].iloc[0])


if __name__ == "__main__":
    main()
