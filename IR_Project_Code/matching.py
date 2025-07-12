import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from create_index import custom_preprocessor
import numpy as np
from nltk.tokenize import word_tokenize
import pinecone
from vector_store_word2vec import get_embedding, get_embedding_tfidf


def get_average_vector(tokens, model, vector_size):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)
    

def compute_doc_vectors(cleaned_dataset_path, word2vec_model, vector_size): 
    df = pd.read_csv(cleaned_dataset_path, usecols=[0, 1])
    df = df.dropna()
    content_column_name = df.columns[1]

    doc_vectors = []
    for content in df[content_column_name]:
        tokens = content.split()  # بافتراض أن التنظيف قد تم مسبقًا
        vector = get_average_vector(tokens, word2vec_model, vector_size=vector_size)
        doc_vectors.append(vector)
    return doc_vectors

def get_query_vector_hybrid(query, vectorizer, w2v_model, svd_model):
    query_clean = custom_preprocessor(query)
    tfidf_part = vectorizer.transform([query_clean])
    tfidf_reduced = svd_model.transform(tfidf_part)  

    tokens = word_tokenize(query_clean)
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if vectors:
        w2v_part = np.mean(vectors, axis=0).reshape(1, -1)
    else:
        w2v_part = np.zeros((1, w2v_model.vector_size))

    return np.hstack([tfidf_reduced.astype(np.float32), w2v_part])


def search(vectorizer, tfidf_matrix, dataset, query, top_n=10):   

    query = custom_preprocessor(query) 
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]    
    top_indices = [i for i in top_indices if similarities[i] > 0.1]
    
    ids_column_name = dataset.columns[0]
    content_column_name = dataset.columns[1]
    top_ids = []
    top_docs = []
    for i in top_indices:

        doc_id = dataset.at[i, ids_column_name]
        content = dataset.at[i, content_column_name]

        top_ids.append(str(doc_id))     
        top_docs.append(str(content))   

    return top_ids, top_docs


def search_word2vec(query, word2vec_model, doc_vectors, dataset, top_n, vector_size):
    
    query_tokens = custom_preprocessor(query).split()
    query_vector = get_average_vector(query_tokens, word2vec_model, vector_size)
    
    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    content_column_name = dataset.columns[1]
    ids_column_name = dataset.columns[0]
    top_indices = [i for i in top_indices if similarities[i] > 0.1]

    top_ids = [str(dataset.at[i, ids_column_name]) for i in top_indices]
    top_docs = [str(dataset.at[i, content_column_name]) for i in top_indices]

    return top_ids, top_docs



def search_hybrid(hybrid_matrix, dataset, query, vectorizer, w2v_model, svd_model, top_n=10):
    query_vector = get_query_vector_hybrid(query, vectorizer, w2v_model, svd_model)
    similarities = cosine_similarity(query_vector, hybrid_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    top_indices = [i for i in top_indices if similarities[i] > 0.1]

    ids_column_name = dataset.columns[0]
    content_column_name = dataset.columns[1]

    top_ids = []
    top_docs = []
    for i in top_indices:

        doc_id = dataset.at[i, ids_column_name]
        content = dataset.at[i, content_column_name]

        top_ids.append(str(doc_id))     
        top_docs.append(str(content)) 

    return top_ids, top_docs


def pinecone_word2vec_search(model, tfidf_vectorizer, dataset, api_key, index_name, query):
    pinecone_client = pinecone.Pinecone(api_key=api_key)
    index = pinecone_client.Index(index_name)
    pr_query = custom_preprocessor(query)

    if index_name == "quora-vector-store-index":
        query_vector = get_embedding(pr_query, model).tolist()
    else:
        query_vector = get_embedding_tfidf(pr_query, model, tfidf_vectorizer).tolist()

    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    
    results = [int(match['id']) for match in results['matches']]
    ids_column_name = dataset.columns[0]
    content_column_name = dataset.columns[1]
    top_ids = []
    top_docs = []
    for i in results:
        top_ids.append(str(dataset.at[i, ids_column_name]))    
        top_docs.append(str(dataset.at[i, content_column_name]))  
    return top_ids, top_docs
