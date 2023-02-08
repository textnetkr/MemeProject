import hydra
from pshmodule.utils import filemanager as fm
import numpy as np
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
        num_labels=cfg.MODEL.num_classes,
    )
    model.eval().cuda()
    print("model loading done!")

    # predict
    df = fm.load(cfg.PATH.origin_class120_ref)
    df_label = fm.load(cfg.PATH.under_label)

    sentence = [
        "말 좀 착하게 하라고",
        "더 넓은 곳에서 살고 싶다",
        "너가 빨래 해놔",
        "너 리얼 별로",
        "스쿼트 인생 기록 찍어야지",
        "말 좀 착하게 하라고",
        "더 넓은 곳에서 살고 싶다",
        "부자되고 싶다 리얼",
    ]

    with torch.no_grad():
        print("--------------------------------------------------------")
        for i in sentence:
            # tokenizer
            data = tokenizer(
                i,
                max_length=cfg.DATASETS.seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # output
            data = {k: v.cuda() for k, v in data.items()}
            outputs = model(**data)

            predict = np.argmax(outputs.logits[0].cpu().numpy())

            # meme extract
            df_ref = df[df.label.values == int(predict)]
            temp_ref = df_ref.sample(frac=1).reset_index(drop=True)
            print(f"🤗 대길이 : {i}")dd
            # print(f"분류 : {df_label[df_label.index.values == int(predict)]}")
            # print(f"유사 문장 : {temp_ref.iloc[0]['u']}")
            print(f"🦝 대춘이 : {temp_ref.iloc[0]['meme']}")
            print(" ")
        print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
