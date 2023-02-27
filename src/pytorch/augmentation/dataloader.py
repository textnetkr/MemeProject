from torch.utils.data import Dataset
from os.path import abspath, splitext
from datasets import load_dataset
from typing import Optional
import pandas as pd

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


class TextStyleTransferDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_length,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index, :].dropna().sample(2)
        # row = self.df.iloc[index, :]
        # text1 = row[0]
        # print(f"text1 : {text1}")
        # row_sample = row.dropna().sample(2)
        # text2 = row_sample[1]
        # print(f"text2 : {text2}")
        text1 = row[0]
        text2 = row[1]
        target_style = row.index[1]
        target_style_name = style_map[target_style]

        encoder_text = f"{text1} [{target_style_name} 문체 변환]"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(
            encoder_text,
            max_length=self.max_length,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                decoder_text,
                max_length=self.max_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        del model_inputs["token_type_ids"]

        return model_inputs


def load(
    tokenizer,
    seq_len,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    # shuffle_seed: Optional[int] = None,
):
    def _tokenize_function(e):
        tokenized = dict()

        temp = []
        temp.append(e)
        df = pd.DataFrame(temp)

        text1 = df["formal"].iloc[0][0]
        row_sample = df.sample()
        print(f"row_sample : {row_sample}")
        print(row_sample.columns)
        text2 = row_sample[0]

        print(f"text1 : {text1}")
        print(f"text2 : {text2}")
        print("-" * 100)

        # row_sample = row.dropna().sample(2)
        # text2 = row_sample[1]
        # print(f"text1 : {text1}")
        # print(f"text2 : {text2}")
        # target_style = row.index[1]
        # target_style_name = style_map[target_style]

        # encoder_text = f"{text1} [{target_style_name} 문체 변환]"
        # decoder_text = f"{text2}{self.tokenizer.eos_token}"

        tokenized = tokenizer(
            e["content"],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        tokenized["labels"] = e["label"]

        return tokenized

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True
    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    # data
    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )
    # if shuffle_seed is not None:
    #     data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        # remove_columns=data["train"].column_names,
    )

    return data["train"], (data["test"] if is_eval else None)


# Write preprocessor code to run in batches.
def default_collator(data):
    return data
