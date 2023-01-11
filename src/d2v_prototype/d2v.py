import base64

import config as cfg
import streamlit as st
from gensim.models.doc2vec import Doc2Vec
from PIL import Image
from pshmodule.utils import filemanager as fm
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor


@st.experimental_singleton
def get_model():
    # data load
    df = fm.load(cfg.temp_data)
    df_ref = fm.load(cfg.origin_ref)

    # doc2vec load
    model = Doc2Vec.load(cfg.doc2vec_soynlp)

    # ltokenizer setting
    # WordExtractor
    word_extractor = WordExtractor(
        min_frequency=10,
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0.0,
    )

    word_extractor.load(cfg.soynlp)
    w = word_extractor.extract()

    cohesion_score = {w: s.cohesion_forward for w, s in w.items()}

    # 사용자 사전 추가
    with open(cfg.user_dict, "r") as f:
        user_dict = f.readlines()
    user_dict = [i.replace("\n", "") for i in user_dict]

    for i in user_dict:
        cohesion_score[i] = 1.0

    # LTokenizer
    tokenizer = LTokenizer(scores=cohesion_score)

    return df, df_ref, model, tokenizer


def main():
    df, df_ref, model, tokenizer = get_model()

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

    set_background("bg1.png")

    # title
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("밈 모범 맛집 대춘이를 만나보세요 (별이 다섯 개 ★★★★★)")

    # 정렬
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
            # vectorize new sentences based on doc2vec
            input_vec = model.infer_vector(tokenizer.tokenize(input))

            # most similarity top 1
            return_docs = model.docvecs.most_similar(
                positive=[input_vec],
                topn=1,
            )
            result_docs = df.u.iloc[int(return_docs[0][0])]
            print(f"유사 문장 : {result_docs}")

            temp_ref = df_ref[df_ref.u.values == result_docs["u"]]
            temp_ref = temp_ref.sample(frac=1).reset_index(drop=True)
            result = str(temp_ref.iloc[0]["meme"])

            # output
            con3 = st.container()
            con3.caption("🤖 대춘이")
            st.info(str(result.strip()))


if __name__ == "__main__":
    main()
