import json
import os
from datasets import load_dataset
from config import CORPUS_DIR

def build_hybrid_corpus(target_passages=50000, output_filename="passages.json"):
    """
    Xây dựng corpus từ SQuAD và Wikipedia (Streaming để chống OOM).
    """
    output_filepath = os.path.join(CORPUS_DIR, output_filename)
    
    if os.path.exists(output_filepath):
        print(f"✅ Corpus đã tồn tại tại {output_filepath}. Không cần tải lại.")
        return output_filepath

    corpus = []
    seen_texts = set()
    current_id = 1

    print("1. Đang tải ngữ cảnh độc nhất từ SQuAD v1.1...")
    squad = load_dataset("squad", split="train")
    for item in squad:
        context = item["context"].strip()
        if context not in seen_texts:
            seen_texts.add(context)
            corpus.append({
                "id": current_id,
                "text": context,
                "title": item["title"]
            })
            current_id += 1
            
    squad_count = len(corpus)
    print(f"   -> Đã trích xuất {squad_count} passages từ SQuAD.")

    print(f"2. Đang streaming Wikipedia để bù đắp cho đủ {target_passages} passages...")
    
    # [ĐÃ FIX BUG Ở ĐÂY]: Dùng wikimedia/wikipedia chuẩn Parquet thay vì script cũ
    wiki_stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    for article in wiki_stream:
        if current_id > target_passages:
            break
            
        paragraphs = [p.strip() for p in article["text"].split("\n\n") if len(p.strip()) > 50]
        
        for p in paragraphs:
            if current_id > target_passages:
                break
            if p not in seen_texts:
                seen_texts.add(p)
                corpus.append({
                    "id": current_id,
                    "text": p,
                    "title": article["title"]
                })
                current_id += 1
                
        if current_id % 10000 == 0:
            print(f"   -> Đã gom được {current_id}/{target_passages} passages...")

    print("3. Đang ghi file xuống Google Drive (I/O Operation)...")
    del seen_texts 
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"✅ HOÀN TẤT! Đã lưu {len(corpus)} passages an toàn tại: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    build_hybrid_corpus()