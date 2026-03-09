import json
import os
import gc
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, RESULTS_DIR
from generator import RAGGenerator
from evaluation import normalize_answer, exact_match_score

def check_groundedness(prediction, context_texts):
    """
    Kiểm tra xem câu trả lời có thực sự nằm trong context không (Context Support).
    Đã xử lý chuẩn hóa để tránh lỗi bắt hụt do viết hoa/dấu câu.
    """
    norm_pred = normalize_answer(prediction)
    if not norm_pred: 
        return False
        
    for c in context_texts:
        if norm_pred in normalize_answer(c):
            return True
    return False

def run_hallucination_analysis(k=3):
    """
    Chạy phân tích lỗi trên cấu hình tốt nhất: Dense Retriever với Top-3.
    """
    test_qa_path = os.path.join(DATA_DIR, "qa", "test_qa.json")
    with open(test_qa_path, "r", encoding="utf-8") as f:
        test_qa = json.load(f)

    # Vẫn dùng 200 câu hỏi như Phase 7 & 8 để so sánh đối chiếu
    import random
    random.seed(42)
    eval_qa = random.sample(test_qa, min(200, len(test_qa)))

    with open(os.path.join(DATA_DIR, "processed", "passages", "chunks.json"), "r") as f:
        chunks = json.load(f)
    
    print("\n[1] Đang khởi tạo mô hình Dense Retriever và Generator...")
    generator = RAGGenerator()
    index = faiss.read_index(os.path.join(DATA_DIR, "embeddings", "faiss_index", "faiss.index"))
    dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=generator.device)

    print(f"\n🚀 ĐANG CHẠY PHÂN TÍCH ẢO GIÁC (DENSE, Top-{k})...")
    
    analysis_log = []
    category_counts = {
        "Correct Grounded": 0,
        "Ungrounded Correct (Leakage)": 0,
        "Context Misinterpretation": 0,
        "True Hallucination": 0
    }
    
    for idx, item in enumerate(eval_qa):
        question = item["question"]
        ground_truths = item["answers"]
        
        # 1. Retrieval
        q_emb = dense_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, top_indices = index.search(q_emb, k)
        retrieved_texts = [chunks[i]["text"] for i in top_indices[0]]
            
        # 2. Generation
        pred_answer = generator.generate_answer(question, retrieved_texts)
        
        # 3. Phân tích trạng thái
        is_correct = exact_match_score(pred_answer, ground_truths) == 1
        is_grounded = check_groundedness(pred_answer, retrieved_texts)
        
        # 4. Phân loại lỗi
        if is_correct and is_grounded:
            error_type = "Correct Grounded"
        elif is_correct and not is_grounded:
            error_type = "Ungrounded Correct (Leakage)"
        elif not is_correct and is_grounded:
            error_type = "Context Misinterpretation"
        else:
            error_type = "True Hallucination"
            
        category_counts[error_type] += 1
        
        # Lưu log chi tiết để làm phụ lục báo cáo
        analysis_log.append({
            "Question": question,
            "Target Answer": ground_truths,
            "Predicted Answer": pred_answer,
            "Error Type": error_type
        })
        
        if (idx + 1) % 50 == 0:
            print(f"   -> Đã phân tích {idx + 1}/{len(eval_qa)} câu...")

    # Dọn dẹp RAM
    del index
    del dense_model
    generator.cleanup()

    # Lưu kết quả thống kê
    summary_df = pd.DataFrame(list(category_counts.items()), columns=["Error Type", "Frequency"])
    summary_df["Percentage (%)"] = (summary_df["Frequency"] / len(eval_qa)) * 100
    
    out_summary_path = os.path.join(RESULTS_DIR, "hallucination_summary.csv")
    summary_df.to_csv(out_summary_path, index=False)
    
    # Lưu log chi tiết
    log_df = pd.DataFrame(analysis_log)
    out_log_path = os.path.join(RESULTS_DIR, "hallucination_detailed_log.csv")
    log_df.to_csv(out_log_path, index=False)

    print(f"\n✅ Đã lưu bảng thống kê tại: {out_summary_path}")
    print(f"✅ Đã lưu log chi tiết từng câu tại: {out_log_path}")
    print("\nBẢNG TỔNG HỢP LỖI:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    pass