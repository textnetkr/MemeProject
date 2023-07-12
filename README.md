# 🤖 MemeProject
밈으로 대답하는 Textual Similarity, Query-Reply Task Model 구현<br><br>

# 👉🏻 model
1. sbert를 활용한 유사도 분석 후 상위 80%에 해당하는 유사 발화에 대한 밈 답변 출력<br>
    유사도 80% 미만이면 LLM을 활용하여 답변 추출
    - command : cd src/sbert/demo
        streamlit run streamlit.py
2. 형태소 분석기 비교<br>
    src/comparison<br>
3. 유사도 기반 밈 대답 봇 with Gensim Doc2vec & Sentence Transformers<br>
    src/d2v_prototype, src/doc2vec<br>
4. 화행, 대화소재 조합으로 koelectra 분류기 시도 코드<br>
    - 조사, 사용자 사전을 활용한 증강 작업<br>
    src/pytorch
5. koalpaca-polyglot5.8b 모델을 활용한 lora 학습 시도 코드<br>
    src/generation_model<br>
<br><br>

# 👉🏻 command
src/doc2vec/preprocessing/memebot_d2v.ipynb<br><br>

# 👉🏻 tree
```bash
.
├── .gitignore
├── README.md
├── requirements.txt
├── src
│   ├── comparison
│   │   ├── add_userdict.ipynb
│   │   ├── add_userdict.ipynb
│   │   └── etri.ipynb
│   ├── d2v_prototype
│   │   ├── d2v.py
│   │   └── score.py
│   ├── doc2vec
│   │   ├── data_augmentation
│   │   │   ├── 1_augmentation.ipynb
│   │   │   ├── 2_doc2vec.ipynb
│   │   │   └── opendict.ipynb
│   │   ├── 1_preprocessing.ipynb
│   │   ├── 2_doc2vec_train.ipynb
│   │   └── 3_memebot_d2v.ipynb
│   ├── generation_model
│   │   ├── custom_lora.py
│   │   ├── dataloader.py
│   │   ├── predict.py
│   │   ├── run.ipynb
│   │   └── train.py
│   ├── pytorch
│   │   ├── data_augmentation
│   │   │   ├── transform
│   │   │   │   ├── predict.py
│   │   │   │   ├── run.ipynb
│   │   │   │   ├── train.py
│   │   │   │   └── train_hf.py
│   │   │   ├── 1.over_sampling.ipynb
│   │   │   ├── 2.userdict.ipynb
│   │   │   ├── 3.chatGPT.ipynb
│   │   │   ├── 4.for_train.ipynb
│   │   │   ├── over_sampling.ipynb
│   │   │   ├── trie.py
│   │   │   └── userdict.ipynb
│   │   ├── generation
│   │   │   ├── run.ipynb
│   │   │   └── userdict.ipynb
│   │   ├── preprocessing
│   │   │   ├── data_check.ipynb
│   │   │   ├── for_gene_train.ipynb
│   │   │   ├── reshape_class.ipynb
│   │   │   └── reshape_class_v2.ipynb
│   │   ├── dataloader.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── sbert
│   │   ├── demo
│   │   │   └── streamlit.py
│   │   ├── preprocessing
│   │   │   └── reshape.ipynb
│   │   ├── predict.py
│   │   ├── ref_emb.py
│   │   ├── run.ipynb
└── └── └── utils.py
```