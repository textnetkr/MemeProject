# import swifter
import config as cfg
from pshmodule.utils import filemanager as fm
from trie import Trie

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# from tqdm import tqdm
# import pandas as pd

kiwi = Kiwi()
stopwords = Stopwords()


def data_load():
    # okt
    def okt(pos):
        result = ""
        if pos in ["NN", "NNP", "NR", "NP"]:
            result = "Noun"  # 명사
        elif pos == "VV":
            result = "Verb"  # 동사
        elif pos == "VA":
            result = "Adjective"  # 형용사
        elif pos == "MM":
            result = "Determiner"  # 관형사
        elif pos in ["MAG", "MAJ"]:
            result = "Adverb"  # 부사
        elif pos == "IC":
            result = "Exclamation"  # 감탄사
        elif pos == "JO":
            result = "Josa"  # 조사
        elif pos == "EO":
            result = "Eomi"  # 어미
        return result

    # mecab
    def mecab(pos):
        result = ""
        if pos == "NN":
            result = "NNG"  # 일반 명사
        elif pos == "NNP":
            result = "NNP"  # 고유 명사
        elif pos == "NR":
            result = "NR"  # 수사
        elif pos == "NP":
            result = "NP"  # 대명사
        elif pos == "VV":
            result = "VV"  # 동사
        elif pos == "VA":
            result = "VA"  # 형용사
        elif pos == "MM":
            result = "MM"  # 관형사
        elif pos == "MAG":
            result = "MAG"  # 일반부사
        elif pos == "MAJ":
            result = "MAJ"  # 접속부사
        elif pos == "IC":
            result = "IC"  # 감탄사
        elif pos == "JO":
            result = "JKS"  # 조사
        return result

    # 발화 데이터
    df = fm.load(cfg.aug_1)
    # 정리
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df = df.iloc[1:]
    # 정렬
    df.sort_values(["g_num", "u_num"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # int→str
    df = df.astype({"g_num": "str", "u_num": "str", "label": "str"})
    df["g_num"] = df.g_num.swifter.apply(lambda x: x.replace(".0", ""))
    df["u_num"] = df.u_num.swifter.apply(lambda x: x.replace(".0", ""))
    df["label"] = df.label.swifter.apply(lambda x: x.replace(".0", ""))

    # 사용자 사전
    user_dict = fm.load(cfg.user_dict)
    # 정리
    user_dict_2 = user_dict[5:]
    user_dict_2.columns = user_dict.iloc[5]
    user_dict_2.reset_index(drop=True, inplace=True)

    user_dict_2 = user_dict_2[1:]
    user_dict_2 = user_dict_2[["표제어", "기본형", "품사", "감정1", "연령", "용례", "반의어"]]
    user_dict_2.rename(
        columns={
            "표제어": "lemma",
            "기본형": "formal",
            "품사": "pos",
            "감정1": "sentiment",
            "연령": "age",
            "용례": "example",
            "반의어": "antonym",
        },
        inplace=True,
    )
    user_dict_2.reset_index(drop=True, inplace=True)
    user_dict_3 = user_dict_2[~user_dict_2.lemma.isnull()]

    user_dict_3["okt_pos"] = user_dict_3.pos.swifter.apply(okt)
    user_dict_3["mecab_pos"] = user_dict_3.pos.swifter.apply(mecab)

    return df, user_dict_3


def trie_dict(dict):
    trie = Trie()
    for d in dict:
        trie.insert(str(d))
    return trie


def kiwi_add(user_dict_3):
    # UserDict Add
    # 일단 사용자 사전을 공백을 제거 후 등록(불변어 기준)
    # 토크나이징 할 때 어떤 문장이 들어오면 공백 제거 후 토큰화
    constant = ["NNG", "NNP", "NR", "NP", "MM", "MAG", "MAJ", "IC"]

    # 합성어 기준
    for c in constant:
        for i in user_dict_3[user_dict_3.mecab_pos == c].iterrows():
            kiwi.add_user_word(str(i[1]["lemma"]).replace(" ", ""), c, 0)

    # def aug():
    """입력 문과 사용자 사전 표제어를 하은 님이 추출해준 실질 형태소&불변어 들만 살린 후 사용자 사전 실질 형태소가 입력 문
        실질 형태소에 있는지를 비교함.
        포함이 있으면 유의어 셋으로 교체하는 방법 - 가변어, 사용자 사전 추가 불변어는 공백 제거로 비교 후 교체(키 밸류로 원본
        들다가 교체 후 뱉기? 확인)
        테스트 : 품사 별로 용례가 입력문으로 가정해서 표제어랑 비교해서 위 상황이 비교될만 한지
        동사, 형용사 등 가변어는 형태소 분석 과정에서 어근 추출을 잘 못하거나 품사를 다르게 추출하는 경우가 생겨서 좀 더 고민이
        필요해 보이고 따라서 불변어 위주로 교체가 우선으로 되려고 함."""


def main():
    # data load
    df, user_dict_3 = data_load()

    # dict to trie
    # trie = trie_dict(user_dict_3.lemma.tolist())

    # kiwi userdict add
    kiwi_add(user_dict_3)

    print(kiwi.tokenize("알잘딱깔센 잘하자"))


if __name__ == "__main__":
    main()
