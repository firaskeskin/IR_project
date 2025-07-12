from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


#--------------------------------------------------------------------------------------------

#((((For Clinical Dataset Training))))

def get_average_vector(tokens, model, vector_size):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)
    

def compute_doc_vectors(cleaned_dataset_path, word2vec_model, vector_size=100):
    df = pd.read_csv(cleaned_dataset_path, usecols=[0, 1])
    df = df.dropna()
    content_column_name = df.columns[1]

    doc_vectors = []
    for content in df[content_column_name]:
        tokens = content.split()
        vector = get_average_vector(tokens, word2vec_model, vector_size)
        doc_vectors.append(vector)

    return doc_vectors


def load_cleaned_data(cleaned_data_path):

    print(f"load cleaning data {cleaned_data_path}...")

    df = pd.read_csv(cleaned_data_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('') 
    df['cleaned_text'] = df['cleaned_text'].astype(str)  
    cleaned_texts = [text.split() for text in df['cleaned_text']]
    return cleaned_texts

def train_word2vec_clincial(cleaned_texts, vector_size=100, window=5, min_count=2, workers=4):
    model = Word2Vec(
        sentences=cleaned_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    model.save("clinical_word2vec.model") 

    print("Word2Vec model trained and saved successfully.")
    return model

cleaned_data_path = "D:/IR_data/data/clinical/cleaned_data.csv" 
cleaned_texts = load_cleaned_data(cleaned_data_path)
model = train_word2vec_clincial(cleaned_texts)
doc_vectors = compute_doc_vectors(cleaned_data_path, model, 100)
doc_vectors_np = np.array(doc_vectors)
np.save("D:/dataset_downloader/clinical_doc_vectors.npy", doc_vectors_np)

print("cl_doc_vectors.npy Saved Successfuly")


#------------------------------------------------------------------------------------------------------------------------

#((((For Quora Dataset Traning))))

def load_cleaned_data(cleaned_data_path):

    print(f"load cleaning data {cleaned_data_path}...")

    df = pd.read_csv(cleaned_data_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
    cleaned_texts = [text.split() for text in df['cleaned_text']]
    return cleaned_texts, df['cleaned_text'].tolist()

def train_word2vec_quora(cleaned_texts, vector_size=100, window=10, min_count=2, workers=4, sg=1):
    model = Word2Vec(
        sentences=cleaned_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg  
    )
    model.save("qu_word2vec.model") 

    print("Word2Vec Trained Successfully")
    return model

def compute_doc_vectors_tfidf(texts, model, vector_size):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    vocab = tfidf.vocabulary_
    
    doc_vectors = []
    for i, text in enumerate(texts):
        tokens = text.split()
        weights = []    
        vectors = []
        for token in tokens:
            if token in model.wv and token in vocab:
                tfidf_weight = tfidf_matrix[i, vocab[token]]
                vectors.append(model.wv[token] * tfidf_weight)
                weights.append(tfidf_weight)
        if vectors:
            doc_vector = np.sum(vectors, axis=0) / (np.sum(weights) + 1e-9)
        else:
            doc_vector = np.zeros(vector_size)
        doc_vectors.append(doc_vector)
    return doc_vectors

cleaned_data_path = "D:/IR_data/data/quora/cleaned_data.csv"
cleaned_tokens, original_texts = load_cleaned_data(cleaned_data_path)
model = train_word2vec_quora(cleaned_tokens, vector_size=100, window=10, sg=1)
doc_vectors = compute_doc_vectors_tfidf(original_texts, model, vector_size=100)

doc_vectors_np = np.array(doc_vectors)
np.save("D:/dataset_downloader/qu_doc_vectors_tfidf_weighted.npy", doc_vectors_np)

print("TF-IDF Data Saved Successfully")
