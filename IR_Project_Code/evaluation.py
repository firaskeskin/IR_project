import csv
from matching import search
import requests
import pandas as pd
from scipy import sparse
import joblib
from tqdm import tqdm
from dataset_cleaner import clean_process_text




# def search(query, dataset):
#     data = {"query": query, "dataset": dataset}
#     response = requests.post("http://127.0.0.1:8001/matching", json=data)
#     return response.json().get("top_ids")


def get_relevant_id_from_qrel(min_rel_val, query_id, csv_file):
    relevant_ids = []
    relevance_scores = {}
    
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            # تأكد أن الصف يحتوي على 3 أعمدة فقط
            if len(row) < 3:
                print(f"❗ Bad row at line {i + 1}: {row}")
                continue
            
            qid, doc_id, rel = row[0], row[1], row[2]
            
            if qid == query_id:
                rel_val = int(rel)
                if rel_val >= min_rel_val:
                    relevant_ids.append(doc_id)
                    relevance_scores[doc_id] = 1
                else:
                    relevance_scores[doc_id] = 0  # نحتفظ بها حتى لا نحذفها من المقارنة لاحقًا

    return relevant_ids, relevance_scores



def precision_at_k(retrieved_docs, relevant_docs, relevance_scores, k):
    relevant_in_top_k = sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs)
    precision = relevant_in_top_k / k
    return precision


def calculate_recall(min_rel_val, retrieved_docs, relevant_docs, relevance_scores):
    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs and relevance_scores[doc] > 0)
    total_relevant = sum(1 for rel in relevance_scores.values() if rel > 0)
    if total_relevant == 0:
        return 0
    recall = relevant_retrieved / total_relevant
    return recall


def average_precision_at_k(retrieved_docs, relevant_docs, relevance_scores, k):
    precision_sum = 0.0
    relevant_count = 0

    for i in range(min(k, len(retrieved_docs))):
        if retrieved_docs[i] in relevant_docs:
            relevant_count += 1
            precision_at_i = precision_at_k(retrieved_docs, relevant_docs, relevance_scores, i + 1)
            precision_sum += (precision_at_i * relevance_scores[retrieved_docs[i]])  

    if relevant_count == 0:
        return 0

    average_precision = precision_sum / relevant_count
    return average_precision


def reciprocal_rank_at_k(min_rel_val, retrieved_docs, relevant_docs, relevance_scores, k=10):
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs and relevance_scores[doc] >= min_rel_val:
            return 1 / (i + 1)
    return 0


def process_queries(vectorizer, tfidf_matrix, dataset, min_rel_val, queries_file, qrels_file, k=10):
    precision_results = []
    recall_results = []
    queries_results = []
    mrr_results = []
    ap_values = []

    df_queries = pd.read_csv(queries_file, encoding='utf-8', usecols=[0, 1], names=["query_id", "text"], header=0)

    for idx, row in df_queries.iterrows():
        try:
            query_id = str(row['query_id']).strip()
            query = str(row['text']).strip()

            if not query_id or not query:
                print(f"Skipping empty row at index {idx}")
                continue
            pr_query = clean_process_text(query)

            top_ids, top_docs = search(vectorizer, tfidf_matrix, dataset, pr_query, k)
            if not top_ids:
                print(f"No results for query: {query_id}")
                continue

            relevant_docs, relevance_scores = get_relevant_id_from_qrel(min_rel_val, query_id, qrels_file)

            precision = precision_at_k(top_ids, relevant_docs, relevance_scores, k)
            recall = calculate_recall(min_rel_val, top_ids, relevant_docs, relevance_scores)
            ap = average_precision_at_k(top_ids, relevant_docs, relevance_scores, k)
            rr = reciprocal_rank_at_k(min_rel_val, top_ids, relevant_docs, relevance_scores, k)

            precision_results.append((query_id, precision * 100))
            recall_results.append((query_id, recall * 100))
            ap_values.append(ap)
            mrr_results.append(rr)

            print(f"Query {query_id} => P@{k}: {precision*100:.2f}%, R@{k}: {recall*100:.2f}%, AP@{k}: {ap*100:.2f}%")

        except Exception as e:
            print(f"Error in query at index {idx} (ID: {row.get('query_id')}): {e}")
            continue

    final_map = sum(ap_values) / len(ap_values) * 100 if ap_values else 0
    final_mrr = sum(mrr_results) / len(mrr_results) * 100 if mrr_results else 0

    return precision_results, recall_results, final_map, final_mrr
        

antique = {
    "dataset": pd.read_csv("D:/IR_data/data/antique/docs.csv", usecols=[0, 1]),
    "vectorizer": joblib.load("D:/dataset_downloader/vectorizer.joblib"),
    "tfidf_matrix": sparse.load_npz("D:/dataset_downloader/tfidf_matrix.npz"),    
}

clinical = {
    "vectorizer": joblib.load("D:/dataset_downloader/cl_vectorizer.joblib"),
    "tfidf_matrix": sparse.load_npz("D:/dataset_downloader/cl_tfidf_matrix.npz"),
    "dataset": pd.read_csv("D:/IR_data/data/clinicaltrials/docs.csv", usecols=[0, 1]),
}

queries_file = 'D:/IR_data/data/clinicaltrials/queries.csv'
qrels_file = 'D:/IR_data/data/clinicaltrials/qrels.csv'

precision_results, recall_results, final_map, final_mrr = process_queries(clinical.get("vectorizer"), clinical.get("tfidf_matrix"), clinical.get("dataset"), 3, queries_file, qrels_file, k=10)

print(f"Final Mean Average Precision (MAP): {final_map:.2f}%")
print(f"Mean Reciprocal Rank (MRR@10): {final_mrr:.2f}%")

                