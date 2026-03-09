import json
import os
from transformers import AutoTokenizer
# [ĐÃ FIX BUG]: Gọi trực tiếp config vì sys.path đã ở thư mục src
from config import CORPUS_DIR, DATA_DIR

def process_chunking(chunk_size=200, overlap=50, input_filename="passages.json", output_filename="chunks.json"):
    """
    Cắt nhỏ văn bản bằng Sliding Window dựa trên Tokenizer của mô hình Embedding.
    """
    input_filepath = os.path.join(CORPUS_DIR, input_filename)
    
    # Thiết lập đường dẫn thư mục processed
    output_dir = os.path.join(DATA_DIR, 'processed', 'passages')
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    if os.path.exists(output_filepath):
        print(f"✅ Chunks đã tồn tại tại {output_filepath}. Bỏ qua bước này.")
        return output_filepath

    print(f"1. Đang tải dữ liệu gốc từ {input_filepath}...")
    with open(input_filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print("2. Đang khởi tạo Tokenizer (all-MiniLM-L6-v2)...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    chunks = []
    print(f"3. Bắt đầu cắt chunk (Size: {chunk_size}, Overlap: {overlap}). Quá trình này có thể mất vài phút...")
    
    for idx, doc in enumerate(corpus):
        tokens = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        
        i = 0
        chunk_idx = 0
        while i < len(tokens) or (i == 0 and len(tokens) == 0):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            if len(chunk_text.strip()) > 0:
                chunks.append({
                    "chunk_id": f"{doc['id']}_{chunk_idx}",
                    "parent_id": doc['id'],
                    "title": doc['title'],
                    "text": chunk_text
                })
            
            i += (chunk_size - overlap)
            chunk_idx += 1
            
        if (idx + 1) % 10000 == 0:
            print(f"   -> Đã xử lý {idx + 1}/{len(corpus)} documents, tạo ra {len(chunks)} chunks tạm thời...")

    print("4. Đang ghi file chunks xuống Google Drive...")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ HOÀN TẤT! Từ {len(corpus)} docs ban đầu đã sinh ra {len(chunks)} chunks.")
    print(f"💾 File lưu tại: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    pass