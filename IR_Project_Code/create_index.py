import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse
from nltk.tokenize import word_tokenize
from dataset_cleaner import clean_process_text

def custom_preprocessor(text):
        return clean_process_text(text)

def get_documents(dataset_path):
    df = pd.read_csv(dataset_path, usecols=[0, 1])
    content_column_name = df.columns[1]

    df = df.dropna(subset=[content_column_name])
    df[content_column_name] = df[content_column_name].astype(str)
    
    documents = list(df[content_column_name])
    return documents


def save_index(vectorizer, tfidf_matrix):
    joblib.dump(vectorizer, "quora(2)_vectorizer.joblib")
    sparse.save_npz("quora(2)_tfidf_matrix.npz", tfidf_matrix)


def create_index_TFID(dataset_path):
    documents = get_documents(dataset_path)
    vectorizer = TfidfVectorizer(max_df=0.95,
                                 min_df=5,
                                 tokenizer=word_tokenize,
                                 preprocessor=custom_preprocessor,
                                 token_pattern=None)


    tfidf_matrix = vectorizer.fit_transform(documents)
    save_index(vectorizer, tfidf_matrix)
    print("Index on antique created and saved successfully.")

