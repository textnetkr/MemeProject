import config as cfg
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from pshmodule.utils import filemanager as fm
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor


def main():
    # data load
    df = fm.load(cfg.temp_data)
    # paraphrase
    df_paraph = fm.load(cfg.paraph)
    print(df_paraph.head())

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

    # doc2vec
    result = []
    for i in df_paraph.iterrows():
        vec = model.infer_vector(tokenizer.tokenize(i[1]["paraphrase"]))

        # most similarity top 1
        return_docs = model.docvecs.most_similar(positive=[vec], topn=1)
        res_docs = df.u.iloc[int(return_docs[0][0])]

        result.append([i[1]["paraphrase"], res_docs["u"], return_docs[0][1]])

    df = pd.DataFrame(result, columns=[["발화문", "유사 문장", "유사도"]])
    fm.save(cfg.score, df)


if __name__ == "__main__":
    main()
