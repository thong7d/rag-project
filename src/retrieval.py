import json
import os
import gc
import numpy as np
import pandas as pd
import faiss
import torch
import random
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from config import DATA_DIR, CORPUS_DIR, RESULTS_DIR

def prepare_test_set(num_samples=1000):
    """
    Giải quyết Tình huống Tạo tập test bằng cách đối chiếu câu hỏi SQuAD 
    với parent_id trong corpus của chúng ta.
    """
    test_qa_path = os.path.join(DATA_DIR, "qa", "test_qa.json")
    os.makedirs(os.path.dirname(test_qa_path), exist_ok=True)
    
    if os.path.exists(test_qa_path):
        print("✅ Tập Test QA đã tồn tại. Đang tải...")
        with open(test_qa_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print("1. Đang tái tạo tập Test QA...")
    with open(os.path.join(CORPUS_DIR, "passages.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    # Tạo bảng tra cứu ngược: Text -> ID
    text_to_id = {doc["text"]: doc["id"] for doc in corpus}
    del corpus
    gc.collect()
    
    squad = load_dataset("squad", split="train")
    valid_qa = []
    
    # Tìm các câu hỏi có context khớp chính xác với corpus của ta
    for item in squad:
        context = item["context"].strip()
        if context in text_to_id:
            valid_qa.append({
                "question": item["question"],
                "target_parent_id": text_to_id[context],
                "answers": item["answers"]["text"] # DÒNG NÀY ĐỂ LẤY ĐÁP ÁN
            })
    # Lấy ngẫu nhiên num_samples câu hỏi
    random.seed(42)
    test_set = random.sample(valid_qa, min(num_samples, len(valid_qa)))
    
    with open(test_qa_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã tạo tập test gồm {len(test_set)} câu hỏi.")
    return test_set

def run_experiments():
    """Thực thi ma trận kiểm thử BM25 và Dense Retrieval."""
    print("-" * 30)
    print("CHUẨN BỊ DỮ LIỆU & MÔ HÌNH")
    print("-" * 30)
    
    test_qa = prepare_test_set(1000)
    
    # Tải Mapping và Chunks
    with open(os.path.join(DATA_DIR, "embeddings", "faiss_index", "chunk_mapping.json"), "r") as f:
        mapping = json.load(f)
        
    with open(os.path.join(DATA_DIR, "processed", "passages", "chunks.json"), "r") as f:
        chunks = json.load(f)
    
    # -----------------------------------------
    # KHỞI TẠO BM25
    # -----------------------------------------
    print("\nĐang khởi tạo BM25 (In-memory)...")
    tokenized_corpus = [chunk["text"].lower().split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    del tokenized_corpus
    gc.collect()
    
    # -----------------------------------------
    # KHỞI TẠO DENSE RETRIEVER
    # -----------------------------------------
    print("Đang khởi tạo FAISS & Dense Model...")
    index = faiss.read_index(os.path.join(DATA_DIR, "embeddings", "faiss_index", "faiss.index"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    results = []
    
    # -----------------------------------------
    # HÀM ĐÁNH GIÁ (EVALUATOR)
    # -----------------------------------------
    def evaluate(retriever_name, k_list):
        print(f"\n🚀 Đang chạy đánh giá cho {retriever_name}...")
        for k in k_list:
            hits = 0
            mrr_sum = 0
            
            for idx, item in enumerate(test_qa):
                question = item["question"]
                target_id = item["target_parent_id"]
                retrieved_parent_ids = []
                
                if retriever_name == "BM25":
                    tokenized_q = question.lower().split(" ")
                    scores = bm25.get_scores(tokenized_q)
                    top_indices = np.argsort(scores)[::-1][:k]
                    retrieved_parent_ids = [mapping[i]["parent_id"] for i in top_indices]
                
                elif retriever_name == "Dense":
                    q_emb = model.encode([question], convert_to_numpy=True)
                    faiss.normalize_L2(q_emb)
                    _, top_indices = index.search(q_emb, k)
                    retrieved_parent_ids = [mapping[i]["parent_id"] for i in top_indices[0]]
                
                # Tính Recall & MRR
                if target_id in retrieved_parent_ids:
                    hits += 1
                    rank = retrieved_parent_ids.index(target_id) + 1
                    mrr_sum += 1.0 / rank
                    
                if (idx + 1) % 250 == 0:
                    print(f"   -> Đã test {idx + 1}/1000 câu hỏi...")
                    
            recall = hits / len(test_qa)
            mrr = mrr_sum / len(test_qa)
            results.append({"Retriever": retriever_name, "k": k, "Recall": recall, "MRR": mrr})
            print(f"   🎯 {retriever_name} (k={k}) -> Recall: {recall:.4f} | MRR: {mrr:.4f}")

    # Chạy ma trận theo roadmap
    evaluate("BM25", [5, 10])
    evaluate("Dense", [5, 10])
    
    # Lưu kết quả
    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "retrieval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu báo cáo thực nghiệm tại: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    pass