import config as cfg
import streamlit as st
from gensim.models.doc2vec import Doc2Vec
from pshmodule.utils import filemanager as fm

df = fm.load(cfg.origin_ref)
model = Doc2Vec.load(cfg.doc2vec_soynlp)


def main():
    print(df.head())
    print(model)
    st.title("MemeBot based on doc2vec similarity - ProtoType")

    input = st.text_input(label="", value="문장을 입력해주세요!")
    print(f"input : {input}")

    if st.button("Confirm"):
        con = st.container()
        con.caption("밈봇 답변 :")
        con.write(input)


if __name__ == "__main__":
    main()
