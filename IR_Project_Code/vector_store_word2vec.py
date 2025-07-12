import numpy as np
import pinecone
from tqdm import tqdm


def get_embedding(text, model):
    tokens = text.split()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def get_embedding_tfidf(text, model, tfidf_vectorizer):
    tokens = text.split()
    vocab = tfidf_vectorizer.vocabulary_
    weights = []
    vectors = []
    tfidf_matrix = tfidf_vectorizer.transform([text])
    
    for token in tokens:
        if token in model.wv and token in vocab:
            weight = tfidf_matrix[0, vocab[token]]
            vectors.append(model.wv[token] * weight)
            weights.append(weight)

    return np.sum(vectors, axis=0) / (np.sum(weights) + 1e-9) if vectors else np.zeros(model.vector_size)

def upload_word2vec_vectors_in_batches(index, doc_vectors, batch_size=100):
    num_docs = len(doc_vectors)
    for start in tqdm(range(0, num_docs, batch_size), desc="📤 جاري رفع المتجهات إلى Pinecone"):
        end = min(start + batch_size, num_docs)
        batch_ids = [str(i) for i in range(start, end)]
        batch_vecs = doc_vectors[start:end]
        
        valid_data = [
            (batch_ids[i], vec.tolist()) 
            for i, vec in enumerate(batch_vecs) 
            if np.any(vec)
        ]
        
        if valid_data:
            index.upsert(valid_data)


def store_doc_vectors(api_key, index_name, doc_vectors, dimension):
    pinecone_client = pinecone.Pinecone(api_key=api_key)
    
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pinecone_client.Index(index_name)
    
    upload_word2vec_vectors_in_batches(index, doc_vectors, batch_size=25)
