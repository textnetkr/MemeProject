import json
import numpy as np
import hydra
import torch
import pandas as pd
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
    # 100,000건 씩 쪼개서
    print(f"data shape : {data.shape}")
    sentences = data.u.tolist()[100001:]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)
    model = AutoModel.from_pretrained(cfg.MODEL.name)

    # Tokenize sentences
    sent_encoded = tokenizer(
        sentences,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    # Compute token embeddings
    with torch.no_grad():
        output = model(**sent_encoded)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_emb = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expand = (
            attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        )
        return torch.sum(token_emb * input_mask_expand, 1) / torch.clamp(
            input_mask_expand.sum(1),
            min=1e-9,
        )

    # Perform pooling. In this case, mean pooling.
    embeddings = mean_pooling(output, sent_encoded["attention_mask"])
    print(f"embeddings : {embeddings[:3]}")
    ref_emb = np.array(embeddings)
    np.save(cfg.PATH.ref_emb2, ref_emb)


if __name__ == "__main__":
    main()
