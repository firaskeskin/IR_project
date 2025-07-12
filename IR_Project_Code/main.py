import joblib
from scipy import sparse
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataset_cleaner import clean_process_text
from create_index import create_index_TFID
import requests
import uvicorn
from matching import search, pinecone_word2vec_search, search_word2vec, search_hybrid
from gensim.models import Word2Vec
from pydantic import BaseModel
from vector_store_word2vec import store_doc_vectors
from query_refinment import apply_query_refinement


api_key = "pcsk_3qzvoy_6QypnV7rTEbQ3rQQvc2Ho96YaHa7ZiuCXeqMwbSH8CZSjTwNyNRFmxuFRGnLBxP"
clinical_index_name = "clinical-vector-store-index"
quora_index_name = "quora-vector-store-index"


app = FastAPI()

clinical = {
    "vectorizer": joblib.load("D:/dataset_downloader/cl_vectorizer.joblib"),
    "tfidf_matrix": sparse.load_npz("D:/dataset_downloader/cl_tfidf_matrix.npz"),
    "dataset": pd.read_csv("D:/IR_data/data/clinicaltrials/docs.csv", usecols=[0, 1]),
    "dataset_path" : "D:/IR_data/data/clinicaltrials/docs.csv",
    "word2vec_model" : Word2Vec.load("cl_word2vec.model"),
    "cleaned_dataset" : "D:/IR_data/data/clinicaltrials/cleaned_data.csv",
    "doc_vectors": np.load("D:/dataset_downloader/cl_doc_vectors.npy"),
    "hybrid_matrix": np.load("D:/dataset_downloader/cl_hybrid_svd_representation.npy"),
    "svd_model" : joblib.load("D:/dataset_downloader/cl_hybrid_svd_model.joblib"),
    "hybrid_vectorizer" : joblib.load("D:/dataset_downloader/cl_hybrid_vectorizer.joblib"),
}

quora = {
    "vectorizer": joblib.load("D:/dataset_downloader/quora(2)_vectorizer.joblib"),
    "tfidf_matrix": sparse.load_npz("D:/dataset_downloader/quora(2)_tfidf_matrix.npz"),
    "dataset": pd.read_csv("D:/IR_data/data/quora/docs.csv", usecols=[0, 1]),
    "dataset_path" : "D:/IR_data/data/quora/docs.csv",
    "word2vec_model" : Word2Vec.load("qu_word2vec.model"),
    "cleaned_dataset" : "D:/IR_data/data/quora/cleaned_data.csv",
    "doc_vectors": np.load("D:/dataset_downloader/qu_doc_vectors_tfidf_weighted.npy"),
    "hybrid_matrix": np.load("D:/dataset_downloader/quora_hybrid_svd_representation.npy"),
    "svd_model" : joblib.load("D:/dataset_downloader/quora_hybrid_svd_model.joblib"),
    "hybrid_vectorizer" : joblib.load("D:/dataset_downloader/quora_hybrid_vectorizer.joblib"),
}
  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextCleanRequest(BaseModel):
    text: str

class IndexingRequest(BaseModel):
    dataset_path: str

class SearchingRequest(BaseModel):        
    query: str
    dataset: str   
    use_pinecone: Optional[bool] = False
    refine: Optional[bool] = False

class VectorInput(BaseModel):
    dataset: str  

class QueryInput(BaseModel):
    query: str
    top_k: int = 5


class RefinementRequest(BaseModel):
    query: str
    representation: str

@app.post("/text-processing")
def process_text(request: TextCleanRequest):
    try:
        processed_text = clean_process_text(request.text)
        return {"processed_text": processed_text}
    except Exception as e:
        return {"message": "something went wrong while processing", "details": str(e)}


@app.post("/indexing")
def indexing(request: IndexingRequest):
    try:
        create_index_TFID(request.dataset_path)
        return {"message": "index created successfully"}
    except Exception as e:
        return {"message": "something went wrong while creating index", "details": str(e)}
    

@app.post("/tfIdf-matching")
def matching(request: SearchingRequest):
    try:
        refined_query = None
        top_ids = []
        top_docs = []

        if request.dataset == "clinical":
            vectorizer = clinical.get("vectorizer")
            tfidf_matrix = clinical.get("tfidf_matrix")
            dataset = clinical.get("dataset")
            representation = "cl_tfidf"

        elif request.dataset == "quora":
            vectorizer = quora.get("vectorizer")
            tfidf_matrix = quora.get("tfidf_matrix")
            dataset = quora.get("dataset")
            representation = "qu_tfidf"

        else:
            return {"message": "Unsupported dataset"}

        if request.refine:
            data = {
                "query": request.query,
                "representation": representation,
                "dataset": request.dataset
            }
            response = requests.post("http://127.0.0.1:8001/query-refinment", json=data)
            refined_query = response.json().get("refined_query")

        top_ids, top_docs = search(vectorizer, tfidf_matrix, dataset, request.query)

        return {
            "query": request.query,
            "refined_query": refined_query,
            "top_ids": top_ids,
            "top_docs": top_docs
        }

    except FileNotFoundError as e:
        return {"message": "file not found", "details": str(e)}
    except Exception as e:
        return {"message": "something went wrong while searching", "details": str(e)}




@app.post("/word2vec-matching")
def matchingWord2Vec(request: SearchingRequest):
    try:
        if request.use_pinecone:
            response = requests.post("http://127.0.0.1:8001/pinecone-search", json={"query": request.query, "dataset": request.dataset})
            return response.json()

        if request.dataset == "clinical":
            model = clinical.get("word2vec_model")
            doc_vectors = clinical.get("doc_vectors")
            dataset = clinical.get("dataset")
            representation = "cl_word2vec"

        elif request.dataset == "quora":
            model = quora.get("word2vec_model")
            doc_vectors = quora.get("doc_vectors")
            dataset = quora.get("dataset")
            representation = "qu_word2vec"

        else:
            return {"message": "Unsupported dataset"}

        refined_query = None
        if request.refine:
            data = {
                "query": request.query,
                "representation": representation,
                "dataset": request.dataset
            }
            response = requests.post("http://127.0.0.1:8001/query-refinment", json=data)
            refined_query = response.json().get("refined_query")

        top_ids, top_docs = search_word2vec(request.query, model, doc_vectors, dataset, 10, 100)

        return {
            "query": request.query,
            "refined_query": refined_query,
            "top_ids": top_ids,
            "top_docs": top_docs
        }
    except FileNotFoundError as e:
        return {"message": "file not found", "details": str(e)}
    except Exception as e:
        return {"message": "something went wrong while searching", "details": str(e)}

        
    

@app.post("/hybrid-matching")
def hybrid_matching(request: SearchingRequest):
    try:
        if request.dataset == "clinical":
            top_ids, top_docs = search_hybrid(
                clinical.get("hybrid_matrix"),
                clinical.get("dataset"),
                request.query,
                clinical.get("hybrid_vectorizer"),
                clinical.get("word2vec_model"),
                clinical.get("svd_model"),
                top_n=10
            )
        elif request.dataset == "quora":
            top_ids, top_docs = search_hybrid(
                quora.get("hybrid_matrix"),
                quora.get("dataset"),
                request.query,
                quora.get("hybrid_vectorizer"),
                quora.get("word2vec_model"),
                quora.get("svd_model"),
                top_n=10
            )

        return {"top_ids": top_ids, "top_docs": top_docs}

    except FileNotFoundError as e:
        return {"message": "file not found", "details": str(e)}
    except Exception as e:
        return {"message": "something went wrong while searching", "details": str(e)} 




@app.post("/upload-document")
def upload_doc_vectors(requset: VectorInput):
    try:
        if requset.dataset == "clinical":
            store_doc_vectors(api_key, clinical_index_name, clinical.get("doc_vectors"), 100)
        elif requset.dataset == "quora":
            store_doc_vectors(api_key, quora_index_name, quora.get("doc_vectors"), 100)
        return {"status": "Doc_Vectors Uploaded"}
    except FileNotFoundError as e:
            return {"message": "file not found", "details": str(e)}
    except Exception as e:
            return {"message": "something went wrong while uploading", "details": str(e)} 


@app.post("/pinecone-search")
def search_docs(request: SearchingRequest):
    try:
        if request.dataset == "clinical":
            top_ids, top_docs = pinecone_word2vec_search(clinical.get("word2vec_model"), None, clinical.get("dataset"), api_key, quora_index_name, request.query)
        elif request.dataset == "quora":
            top_ids, top_docs = pinecone_word2vec_search(quora.get("word2vec_model"), quora.get("vectorizer"), quora.get("dataset"), api_key, clinical_index_name, request.query)

        return {"top_ids": top_ids, "top_docs": top_docs}
    except FileNotFoundError as e:
        return {"message": "file not found", "details": str(e)}
    except Exception as e:
        return {"message": "something went wrong while searching", "details": str(e)} 


@app.post("/query-refinment")
def refine_query(request: RefinementRequest):
    refined = apply_query_refinement(request.query ,request.representation)
    return {"refined_query": refined}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
