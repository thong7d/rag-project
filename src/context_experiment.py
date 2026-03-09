import json
import os
import gc
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, RESULTS_DIR
from generator import RAGGenerator
from evaluation import exact_match_score, f1_score

def run_compression_experiment():
    """
    Thực thi Phase 8: Context Compression.
    So sánh độ chính xác khi dùng Top-1, Top-3, Top-5 passages.
    """
    test_qa_path = os.path.join(DATA_DIR, "qa", "test_qa.json")
    with open(test_qa_path, "r", encoding="utf-8") as f:
        test_qa = json.load(f)

    # Dùng 200 câu hỏi giống Phase 7 để đảm bảo tính công bằng (Apple-to-Apple comparison)
    import random
    random.seed(42)
    eval_qa = random.sample(test_qa, min(200, len(test_qa)))

    with open(os.path.join(DATA_DIR, "processed", "passages", "chunks.json"), "r") as f:
        chunks = json.load(f)
    
    print("\n[1] Đang khởi tạo mô hình Dense Retriever và Generator...")
    generator = RAGGenerator()
    index = faiss.read_index(os.path.join(DATA_DIR, "embeddings", "faiss_index", "faiss.index"))
    dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=generator.device)

    results = []
    out_path = os.path.join(RESULTS_DIR, "compression_results.csv")
    
    # Cấu hình thử nghiệm số lượng đoạn văn
    context_sizes = [1, 3, 5]

    for k in context_sizes:
        print(f"\n🚀 ĐANG CHẠY THÍ NGHIỆM COMPRESSION | Số passages: {k}")
        
        em_total = 0
        f1_total = 0
        
        for idx, item in enumerate(eval_qa):
            question = item["question"]
            ground_truths = item["answers"]
            
            # Retrieval (Dense)
            q_emb = dense_model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            _, top_indices = index.search(q_emb, k)
            retrieved_texts = [chunks[i]["text"] for i in top_indices[0]]
                
            # Generation
            pred_answer = generator.generate_answer(question, retrieved_texts)
            
            # Đánh giá
            em = exact_match_score(pred_answer, ground_truths)
            f1 = f1_score(pred_answer, ground_truths)
            em_total += em
            f1_total += f1
            
            if (idx + 1) % 50 == 0:
                print(f"   -> Đã chạy {idx + 1}/{len(eval_qa)} câu. Tạm tính EM: {em_total/(idx+1):.4f}")

        avg_em = em_total / len(eval_qa)
        avg_f1 = f1_total / len(eval_qa)
        
        # Giả định trung bình mỗi đoạn văn là 200 tokens (từ Phase 3)
        estimated_tokens = k * 200
        
        results.append({
            "Context_Size (Passages)": k, 
            "Estimated_Tokens": estimated_tokens,
            "EM": avg_em, 
            "F1": avg_f1
        })
        print(f"🎯 KẾT QUẢ Top-{k}: EM = {avg_em:.4f} | F1 = {avg_f1:.4f}")

    # Dọn dẹp RAM
    del index
    del dense_model
    generator.cleanup()

    # Lưu kết quả
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu báo cáo Compression tại: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    pass