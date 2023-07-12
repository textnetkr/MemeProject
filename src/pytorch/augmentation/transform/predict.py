import hydra
from pshmodule.utils import filemanager as fm
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    # tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.name)

    # dataload
    df = pd.read_csv(cfg.PATH.dataset, sep="\t")
    row_notna_count = df.notna().sum(axis=1)
    row_notna_count.plot.hist(bins=row_notna_count.max())
    df = df[row_notna_count >= 2]

    df_arg = fm.load(cfg.PATH.arg_aug_data)

    # predict
    nlg_pipeline = pipeline(
        "text2text-generation",
        model=cfg.TRAININGARGS.output_dir,
        tokenizer=cfg.MODEL.name,
    )

    style_map = {
        "formal": "문어체",
        "informal": "구어체",
        "android": "안드로이드",
        "azae": "아재",
        "chat": "채팅",
        "choding": "초등학생",
        "emoticon": "이모티콘",
        "enfp": "enfp",
        "gentle": "신사",
        "halbae": "할아버지",
        "halmae": "할머니",
        "joongding": "중학생",
        "king": "왕",
        "naruto": "나루토",
        "seonbi": "선비",
        "sosim": "소심한",
        "translator": "번역기",
    }

    def generate_text(pipe, text, target_style, num_return_seq=5, len=60):
        target_style_name = style_map[target_style]
        text = f"{text} [{target_style_name} 문체 변환]"
        out = pipe(text, num_return_sequences=num_return_seq, max_length=len)
        return [x["generated_text"] for x in out]

    df_transform = []

    for i in tqdm(df_arg.iterrows()):
        print(" ")
        print(f"입력 문장 : {i[1]['u']}")
        for style in df.columns:
            print(
                style,
                generate_text(
                    nlg_pipeline,
                    i,
                    style,
                    num_return_sequences=cfg.DATASET.num_return_sequences,
                    max_length=cfg.DATASET.max_length,
                )[0],
            )
            df_transform.append(
                [
                    i[1]["g_num"],
                    i[1]["u_num"],
                    i[1]["material"],
                    i[1]["speech"],
                    generate_text(
                        nlg_pipeline,
                        i[1]["u"],
                        style,
                        num_return_sequences=1,
                        max_length=1000,
                    )[0],
                    style,
                    i[1]["arg"],
                    i[1]["label"],
                ]
            )


if __name__ == "__main__":
    main()
