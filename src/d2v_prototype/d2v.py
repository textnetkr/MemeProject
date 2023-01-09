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

    # title
    st.title("doc2vec 유사도 기반 밈봇")
    st.header("")
    st.header("")
    st.header("")
    # image
    image = Image.open("bot.png")
    st.image(
        image,
        caption=None,
        width=50,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    # input text
    input = st.text_input(
        label="",
        placeholder="대춘이에게 말을 걸어주세요!",
    )
    print(f"input : {input}")

    if st.button("Enter"):
        st.header("")
        # vectorize new sentences based on doc2vec
        input_vec = model.infer_vector(tokenizer.tokenize(input))

        # most similarity top 1
        return_docs = model.docvecs.most_similar(positive=[input_vec], topn=1)
        result_docs = df.u.iloc[int(return_docs[0][0])]
        print(f"유사 문장 : {result_docs}")

        temp_ref = df_ref[df_ref.u.values == result_docs["u"]]
        temp_ref = temp_ref.sample(frac=1).reset_index(drop=True)
        result = str(temp_ref.iloc[0]["meme"])

        # output
        con = st.container()
        con.caption("밈봇 답변")
        st.success(str(result.strip()))


if __name__ == "__main__":
    main()
