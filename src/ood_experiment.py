import os
import gc
import numpy as np
import pandas as pd
import faiss
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import RESULTS_DIR
from generator import RAGGenerator
from evaluation import exact_match_score, f1_score

def run_ood_experiment(k=5, sample_size=100):
    print("1. Đang tải tập dữ liệu Out-of-Domain (SciQ)...")
    # Dataset khoa học, khác biệt hoàn toàn với bách khoa toàn thư SQuAD
    sciq = load_dataset("sciq", split="test")
    
    # Lọc các câu có context (support)
    valid_data = [item for item in sciq if item["support"].strip() != ""]
    import random
    random.seed(42)
    eval_qa = random.sample(valid_data, min(sample_size, len(valid_data)))
    
    # Tạo mini-corpus từ tập test này
    ood_corpus = list(set([item["support"] for item in valid_data]))
    print(f"   -> Đã tạo Mini OOD Corpus với {len(ood_corpus)} passages độc nhất.")

    generator = RAGGenerator()
    results = []
    
    out_path = os.path.join(RESULTS_DIR, "ood_generation_results.csv")

    for retriever_type in ["BM25", "Dense"]:
        print(f"\n🚀 ĐANG CHẠY OOD SETTING: {retriever_type} | Top-k: {k}")
        
        if retriever_type == "BM25":
            tokenized_corpus = [c.lower().split() for c in ood_corpus]
            retriever = BM25Okapi(tokenized_corpus)
        else:
            dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=generator.device)
            embeddings = dense_model.encode(ood_corpus, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

        em_total = 0
        f1_total = 0
        
        for idx, item in enumerate(eval_qa):
            question = item["question"]
            ground_truth = [item["correct_answer"]]
            
            if retriever_type == "BM25":
                tokenized_q = question.lower().split()
                scores = retriever.get_scores(tokenized_q)
                top_indices = np.argsort(scores)[::-1][:k]
                retrieved_texts = [ood_corpus[i] for i in top_indices]
            else:
                q_emb = dense_model.encode([question], convert_to_numpy=True)
                faiss.normalize_L2(q_emb)
                _, top_indices = index.search(q_emb, k)
                retrieved_texts = [ood_corpus[i] for i in top_indices[0]]
                
            pred_answer = generator.generate_answer(question, retrieved_texts)
            
            em_total += exact_match_score(pred_answer, ground_truth)
            f1_total += f1_score(pred_answer, ground_truth)
            
            if (idx + 1) % 25 == 0:
                print(f"   -> Đã chạy {idx + 1}/{len(eval_qa)} câu. Tạm tính EM: {em_total/(idx+1):.4f}")

        avg_em = em_total / len(eval_qa)
        avg_f1 = f1_total / len(eval_qa)
        results.append({"Retriever": retriever_type, "k": k, "EM": avg_em, "F1": avg_f1})
        print(f"🎯 KẾT QUẢ OOD {retriever_type}(k={k}): EM = {avg_em:.4f} | F1 = {avg_f1:.4f}")
        
        if retriever_type == "Dense":
            del index
            del dense_model
        else:
            del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generator.cleanup()
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu báo cáo OOD tại: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    pass