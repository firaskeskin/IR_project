import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm


df_clean = pd.read_csv("D:/IR_data/data/quora/cleaned_data.csv")
documents = df_clean["cleaned_text"].astype(str).tolist()

max_vocab_size = 20000
vectorizer = TfidfVectorizer(max_features=max_vocab_size)
tfidf_matrix = vectorizer.fit_transform(documents) 

reduced_dimensions = 300
svd = TruncatedSVD(n_components=reduced_dimensions)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

joblib.dump(vectorizer, "quora_hybrid_vectorizer.joblib")
joblib.dump(svd, "quora_hybrid_svd_model.joblib")
print("SVD and Vectorizer Save Successfuly")

w2v_model = Word2Vec.load("qu_word2vec.model")

def get_w2v_vector(text):
    tokens = word_tokenize(text)
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

print("Generating Word2Vec vectors...")
w2v_vectors = np.array([get_w2v_vector(doc) for doc in tqdm(documents)], dtype=np.float32)

tfidf_reduced = tfidf_reduced.astype(np.float32)
hybrid_vectors = np.hstack([tfidf_reduced, w2v_vectors])

np.save("quora_hybrid_svd_representation.npy", hybrid_vectors)

print("Hybrid Representation Saved Successfuly : quora_hybrid_svd_representation.npy")
