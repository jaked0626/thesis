import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Text preprocessiong
import nltk
# nltk.download('stopwords')
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# wn = nltk.WordNetLemmatizer()
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP
import MeCab
from dataclasses import dataclass
import numpy as np
import gensim

NEWS_PATH = "./data/news/{}.csv"

@dataclass 
class TopicModelBert:
    model: BERTopic
    topics: list
    probs: np.array
    doc_info: pd.DataFrame
    topic_features: dict


def load_articles():
    newspapers = ["mainichi", "yomiuri", "asahi", "nikkei"]
    total_df = pd.DataFrame()
    for newspaper in newspapers:
        newspaper_df = pd.read_csv(NEWS_PATH.format(newspaper))
        newspaper_df["publication"] = newspaper
        total_df =  total_df.append(newspaper_df, ignore_index = True)

    total_df = total_df.sort_values(by="date", ignore_index=True)

    # save to csv
    total_df.to_csv(NEWS_PATH.format("articles"), index=False)

    
    return total_df

def parse(tweet_temp):
    t = MeCab.Tagger()
    temp1 = t.parse(tweet_temp)
    temp2 = temp1.split("\n")
    print(temp1)
    print(temp2)
    t_list = []
    for keitaiso in temp2:
        if keitaiso not in ["EOS",""]:
            word,hinshi = keitaiso.split("\t")
            t_temp = [word]+hinshi.split(",")
            if len(t_temp) != 10:
                t_temp += ["*"]*(10 - len(t_temp))
            t_list.append(t_temp)

    return t_list

def parse_to_df(tweet_temp):
    return pd.DataFrame(parse(tweet_temp),
                        columns=["単語","品詞","品詞細分類1",
                                 "品詞細分類2","品詞細分類3",
                                 "活用型","活用形","原形","読み","発音"])

def make_lda_docs(texts: list[str], lemmatize: bool = False) -> list:
    docs = []
    for text in texts:
        df = parse_to_df(text)
        df = df[df["品詞"].isin(["名詞", "動詞"])]
        df = df[df["品詞細分類1"] != "固有名詞"]
        if lemmatize:
            doc = " ".join(df["原形"]).replace("*", "")
        else:
            doc = " ".join(df["単語"])
        docs.append(doc)
    
    return docs

def perform_lda(texts: list[str]) -> gensim.models.LdaModel:
    # docs = articles["title"] + articles["text"]
    docs = [doc.split() for doc in make_lda_docs(texts)]
    dictionary = gensim.corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print(f"Example of original text:\n {texts[0][:200]}")
    print(f"Parsed for LDA:\n {','.join(docs[0])}")

    # perform lda 
    n_cluster = 6
    lda = gensim.models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=n_cluster, 
                    minimum_probability=0.001,
                    passes=20, 
                    update_every=0, 
                    chunksize=10000,
                    random_state=1
                    )
                    
    return lda


def add_unique_publication_count(articles: pd.DataFrame) -> pd.DataFrame:
    # add indicator for number of unique publications on that date 
    unique_publications = articles.groupby(by=['date'])['publication'].nunique().rename('unique_publications')
    articles = articles.merge(unique_publications, on="date", how="left")

    return articles 

def topic_modeling_bert(docs: iter, umap: UMAP = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')) -> TopicModelBert:
    model = BERTopic(language="japanese", calculate_probabilities=True, verbose=True, nr_topics = "auto", umap_model=umap) # 
    topics, probs = model.fit_transform(docs)
    doc_info = model.get_document_info(docs)
    topic_features = model.topic_representations_
    topic_model_bert = TopicModelBert(model=model, topics=topics, probs=probs, doc_info=doc_info, topic_features=topic_features)
    return topic_model_bert 


def main():
    load_articles()


if __name__ == "__main__":
    main()






