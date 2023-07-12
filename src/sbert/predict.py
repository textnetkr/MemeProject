import json
import hydra
import numpy as np
import pandas as pd
import torch
from utils import mean_pooling
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModel


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # data load
    def data_load(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        data = pd.DataFrame(data)
        return data

    data = data_load(cfg.PATH.ref_data)

    # ref load
    ref_emb = np.load(cfg.PATH.ref_emb + ".npy")
    ref_emb = ref_emb.tolist()

    input = ["너... 진짜 짜증나게 하지마라"]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)
    model = AutoModel.from_pretrained(cfg.MODEL.name)

    # Tokenize sentences
    sent_encoded = tokenizer(
        input,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    # Compute token embeddings
    with torch.no_grad():
        output = model(**sent_encoded)

    # Perform pooling. In this case, mean pooling.
    input_emb = mean_pooling(output, sent_encoded["attention_mask"])

    # cosine similarity
    cosine_similarities = linear_kernel(input_emb, ref_emb).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]

    print(f"코사인 유사도 연산 결과 : {related_docs_indices}")
    print(f"입력 문장 : {input[0]}")
    print(f"유사 문장 : {data.iloc[related_docs_indices[0]]['u']}")
    print(f"밈 답변 : {data.iloc[related_docs_indices[0]]['meme']}")


if __name__ == "__main__":
    main()
