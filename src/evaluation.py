import json
import os
import gc
import re
import string
import collections
import numpy as np
import pandas as pd
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, RESULTS_DIR
from generator import RAGGenerator

# ==========================================
# CÁC HÀM METRIC CHUẨN CỦA SQuAD v1.1
# ==========================================
def normalize_answer(s):
    """Chuẩn hóa văn bản: in thường, xóa dấu câu, mạo từ và khoảng trắng thừa."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truths):
    """Tính EM: 1 nếu trùng khớp hoàn toàn với ít nhất 1 đáp án gốc."""
    return int(any(normalize_answer(prediction) == normalize_answer(gt) for gt in ground_truths))

def f1_score(prediction, ground_truths):
    """Tính F1 token-level."""
    def compute_f1(pred, gt):
        pred_tokens = normalize_answer(pred).split()
        gt_tokens = normalize_answer(gt).split()
        common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)
    
    return max([compute_f1(prediction, gt) for gt in ground_truths])

# ==========================================
# PIPELINE THỰC NGHIỆM GENERATION
# ==========================================
def run_generation_experiments():
    test_qa_path = os.path.join(DATA_DIR, "qa", "test_qa.json")
    with open(test_qa_path, "r", encoding="utf-8") as f:
        test_qa = json.load(f)
        
    if "answers" not in test_qa[0]:
        raise ValueError("TÌNH HUỐNG A: Tập test chưa có đáp án! Hãy xóa file test_qa.json và chạy lại prepare_test_set().")

    print(f"Đang tải {len(test_qa)} câu hỏi test...")
    
    # Giảm số lượng xuống 200 câu ngẫu nhiên để tránh timeout Colab (200 câu là đủ ý nghĩa thống kê cho đồ án)
    import random
    random.seed(42)
    eval_qa = random.sample(test_qa, min(200, len(test_qa))) 

    # Load dữ liệu corpus để BM25 truy xuất text
    with open(os.path.join(DATA_DIR, "processed", "passages", "chunks.json"), "r") as f:
        chunks = json.load(f)
    
    with open(os.path.join(DATA_DIR, "embeddings", "faiss_index", "chunk_mapping.json"), "r") as f:
        mapping = json.load(f)
        
    print("\n[1] Đang khởi tạo Generator (Flan-T5)...")
    generator = RAGGenerator()
    
    results = []
    out_path = os.path.join(RESULTS_DIR, "generation_results.csv")
    
    # Ma trận thực nghiệm theo Roadmap
    settings = [
        {"retriever": "BM25", "k": 5},
        {"retriever": "Dense", "k": 5},
        {"retriever": "Dense", "k": 10}
    ]

    for setting in settings:
        retriever_type = setting["retriever"]
        k = setting["k"]
        print(f"\n🚀 ĐANG CHẠY SETTING: {retriever_type} | Top-k: {k}")
        
        # Load Retriever cục bộ để tối ưu RAM
        if retriever_type == "BM25":
            tokenized_corpus = [c["text"].lower().split(" ") for c in chunks]
            retriever = BM25Okapi(tokenized_corpus)
            del tokenized_corpus
            gc.collect()
        else:
            index = faiss.read_index(os.path.join(DATA_DIR, "embeddings", "faiss_index", "faiss.index"))
            dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=generator.device)

        em_total = 0
        f1_total = 0
        
        for idx, item in enumerate(eval_qa):
            question = item["question"]
            ground_truths = item["answers"]
            
            # Retrieval Step
            retrieved_texts = []
            if retriever_type == "BM25":
                tokenized_q = question.lower().split(" ")
                scores = retriever.get_scores(tokenized_q)
                top_indices = np.argsort(scores)[::-1][:k]
                retrieved_texts = [chunks[i]["text"] for i in top_indices]
            else:
                q_emb = dense_model.encode([question], convert_to_numpy=True)
                faiss.normalize_L2(q_emb)
                _, top_indices = index.search(q_emb, k)
                retrieved_texts = [chunks[i]["text"] for i in top_indices[0]]
                
            # Generation Step
            pred_answer = generator.generate_answer(question, retrieved_texts)
            
            # Evaluation Step
            em = exact_match_score(pred_answer, ground_truths)
            f1 = f1_score(pred_answer, ground_truths)
            em_total += em
            f1_total += f1
            
            if (idx + 1) % 50 == 0:
                print(f"   -> Đã chạy {idx + 1}/{len(eval_qa)} câu. Tạm tính EM: {em_total/(idx+1):.4f}")

        # Tính trung bình cho setting này
        avg_em = em_total / len(eval_qa)
        avg_f1 = f1_total / len(eval_qa)
        results.append({"Retriever": retriever_type, "k": k, "EM": avg_em, "F1": avg_f1})
        print(f"🎯 KẾT QUẢ {retriever_type}(k={k}): EM = {avg_em:.4f} | F1 = {avg_f1:.4f}")
        
        # Dọn dẹp RAM cục bộ của retriever này
        if retriever_type == "Dense":
            del index
            del dense_model
        else:
            del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Lưu kết quả tổng hợp
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu báo cáo Generation tại: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    pass