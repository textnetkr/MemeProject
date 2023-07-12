import os
import base64
from PIL import Image
import json
import config as cfg
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from hugchat import hugchat
from hugchat.login import Login


# streamlit run streamlit.py
# @st.experimental_singleton()
@st.cache_resource()
def get_model():
    # data load
    def data_load(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        data = pd.DataFrame(data)
        return data

    # Load model from HuggingFace Hub
    model = SentenceTransformer(cfg.model)

    # reference data
    df_ref = data_load(cfg.ref_data)

    # reference embeddings
    emb_list = os.listdir(cfg.emb_list)
    emb_list = sorted([i for i in emb_list if ".npy" in i])

    print("file load start")
    ref_embs = [np.load(cfg.emb_list + i).tolist() for i in tqdm(emb_list)]
    print("file load end")
    ref_emb = torch.Tensor(sum(ref_embs, []))
    print("tensor end")
    print(ref_emb.shape)

    # HuggingChat
    sign = Login(cfg.email, cfg.passwd)
    cookies = sign.login()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    # Google Translate
    trans = Translator()

    return df_ref, ref_emb, model, chatbot, trans


def main():
    df_ref, ref_emb, model, chatbot, trans = get_model()

    # background image
    def get_base64(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = (
            """
            <style>
            .stApp {
            margin: 0 auto;
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            background-color: #fbe100;
            }
            </style>
            """
            % bin_str
        )
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background("bg.png")

    # title
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("밈 모범 맛집 대춘이를 만나보세요 (별이 다섯 개 ★★★★★)")

    # align
    mystyle = """
    <style>
        p {
            text-align: justify;
        }
    </style>
    """
    st.markdown(mystyle, unsafe_allow_html=True)

    # Generate Two equal columns
    c1, c2 = st.columns((1, 1))
    c11, c22 = st.columns((1, 1))
    c111, c222 = st.columns((1, 1))

    # 대춘이
    with c1:
        con1 = st.container()
        con1.caption("🤖 대춘이")
        # image
        image = Image.open("daechoon.png")
        st.image(
            image,
            caption=None,
            width=150,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )
        con2 = st.container()
        con2.info("대춘이에게 말을 걸어주세요!")

    # My Utterance
    with c22:
        # input text
        input = st.text_input(label="🤗 대길이")
        print(f"input : {input}")
    with c111:
        if input:
            # Tokenize sentences
            query_emb = model.encode([input], convert_to_tensor=True)
            hits = util.semantic_search(query_emb, ref_emb, top_k=1)

            # output
            con3 = st.container()
            con3.caption("🤖 대춘이")
            score = round(hits[0][0]["score"], 2) * 100
            print(f"score :{score}")
            if score >= 80:
                print(f'유사 문장 : {df_ref.iloc[hits[0][0]["corpus_id"]]["u"].strip()}')
                df_ref = (
                    df_ref[df_ref.u == df_ref.iloc[hits[0][0]["corpus_id"]]["u"]]
                    .sample(frac=1)
                    .reset_index(drop=True)
                )
                print(f"밈 대답 : {df_ref.iloc[0]['meme']}")
                st.success(f"{df_ref.iloc[0]['meme'].strip()}", icon="⭐️")
                # st.info(f"⭐️밈 답변⭐️ : {df_ref.iloc[0]['meme'].strip()}")
            else:
                # Google Translate
                google = Translator()
                trans = google.translate(f"{input} 일상 대화로 반말로 대답해줘.", dest="en")
                # HuggingChat
                print(chatbot.chat(trans.text))
                result = google.translate(chatbot.chat(trans.text), dest="ko")  # 초록 번역
                print(result.text)
                st.info(
                    result.text.replace("오픈 어시스턴트", "대춘이").replace(
                        "Open Assistant", "대춘이"
                    )
                )


if __name__ == "__main__":
    main()
