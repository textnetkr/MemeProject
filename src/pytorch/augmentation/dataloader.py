from torch.utils.data import Dataset


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
        text1 = row[0]
        text2 = row[1]
        target_style = row.index[1]
        target_style_name = style_map[target_style]

        encoder_text = f"{text1} [{target_style_name} 문체 변환]"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(
            encoder_text, max_length=self.max_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                decoder_text, max_length=self.max_length, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        del model_inputs["token_type_ids"]

        return model_inputs
