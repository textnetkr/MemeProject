import json
import time
import numpy as np
import hydra
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer


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

    df = data_load(cfg.PATH.ref_data)
    print(df.shape)

    # Load model from HuggingFace Hub
    model = SentenceTransformer(cfg.MODEL.name)

    # 100,000건 씩 쪼개서 embedding
    result = []
    count = 100000
    start = time.time()
    chunks = [df[i : i + count] for i in range(0, df.shape[0], count)]

    for i, chunk in tqdm(enumerate(chunks)):
        sentences = chunk.u.tolist()
        print(f"{i}번째")
        print(f"문장 길이 : {len(sentences)}")

        embeddings = model.encode(sentences)
        result.append(embeddings.tolist())
        print(f"embeddings : {embeddings[:3]}")

        # time taken
        ckpt = time.time()
        print(f"걸린 시간 : {time.strftime('%Y-%m-%d %H:%M:%S')}, {ckpt-start}")

        end = time.time()
        print(f"종료 시간 : {time.strftime('%Y-%m-%d %H:%M:%S')}, {end-start}")
        print("save start")

        # tensor to list
        ref_result = np.array(sum(result, []))
        print(len(ref_result))
        print(ref_result)
        np.save(cfg.PATH.ref_emb + f"_{i+1}", ref_result)
        print("save done")
        result = []


if __name__ == "__main__":
    main()
