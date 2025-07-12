import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import joblib


nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def expand_with_word2vec(model, query: str, topn: int = 3) -> str:
    tokens = word_tokenize(query.lower())
    filtered_tokens = [token for token in tokens if token.isalpha and token not in stop_words]

    expanded_tokens = set(filtered_tokens)

    for token in filtered_tokens:
        if token in model.wv.key_to_index:
            similar_words = model.wv.most_similar(token, topn=topn)
            for word, _ in similar_words:
                expanded_tokens.add(word)

    return " ".join(expanded_tokens)


def reweight_terms_tfidf(query, vectorizer, topn: int = 5) -> str:
    vocab = vectorizer.vocabulary_
    tokens = query.lower().split()
    weighted_terms = []

    for token in tokens:
        if token in vocab:
            try:
                token_index = vectorizer.vocabulary_[token]
                weight = vectorizer.idf_[token_index]
                weighted_terms.append((token, weight))
            except:
                continue

    weighted_terms.sort(key=lambda x: x[1], reverse=True)
    top_terms = [term for term, _ in weighted_terms[:topn]]

    return " ".join(top_terms)


def apply_query_refinement(query, representation: str) -> str:
    if (representation == "cl_word2vec") :
        model = Word2Vec.load("cl_word2vec.model")
        return expand_with_word2vec(model, query)
    
    elif (representation == "qu_word2vec"):
        model = Word2Vec.load("qu_word2vec.model")
        return expand_with_word2vec(model, query)
    
    elif (representation == "cl_tfIdf"):
        vectorizer  = joblib.load("D:/dataset_downloader/cl_vectorizer.joblib")
        return reweight_terms_tfidf(query, vectorizer)
    
    elif (representation == "qu_tfIdf"):
        vectorizer  = joblib.load("D:/dataset_downloader/quora(2)_vectorizer.joblib")
        return reweight_terms_tfidf(query, vectorizer)
    else:
        return query
