import json
import os
import numpy as np
import faiss
import torch
import gc
from sentence_transformers import SentenceTransformer
from config import DATA_DIR

def process_embedding(batch_size=64, input_filename="chunks.json"):
    """
    Chuyển đổi văn bản thành vector nhúng và xây dựng FAISS Index.
    """
    input_filepath = os.path.join(DATA_DIR, 'processed', 'passages', input_filename)
    index_dir = os.path.join(DATA_DIR, 'embeddings', 'faiss_index')
    os.makedirs(index_dir, exist_ok=True)
    
    faiss_path = os.path.join(index_dir, "faiss.index")
    embeddings_path = os.path.join(index_dir, "passage_embeddings.npy")
    mapping_path = os.path.join(index_dir, "chunk_mapping.json")

    if os.path.exists(faiss_path) and os.path.exists(embeddings_path):
        print(f"✅ Index đã tồn tại tại {index_dir}. Bỏ qua bước này.")
        return faiss_path, mapping_path

    print(f"1. Đang tải {input_filepath} vào bộ nhớ...")
    with open(input_filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Tách data để giải phóng RAM cấu trúc JSON phức tạp
    texts = [chunk["text"] for chunk in chunks]
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "parent_id": chunk["parent_id"]} for chunk in chunks]
    
    # Giải phóng biến chunks gốc để tránh OOM
    del chunks
    gc.collect()

    print("2. Đang khởi tạo mô hình all-MiniLM-L6-v2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> Đang chạy trên thiết bị: {device.upper()}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print(f"3. Bắt đầu embedding {len(texts)} chunks (Batch size: {batch_size})...")
    # Encode tự động chia batch, hiển thị thanh tiến trình
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    print("4. Đang chuẩn hóa vector (L2 Normalization) để dùng Cosine Similarity...")
    faiss.normalize_L2(embeddings)

    print("5. Đang xây dựng FAISS Index (IndexFlatIP)...")
    dimension = embeddings.shape[1] # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("6. Đang lưu kết quả xuống Google Drive...")
    # Lưu Index
    faiss.write_index(index, faiss_path)
    # Lưu Numpy Array nguyên gốc
    np.save(embeddings_path, embeddings)
    # Lưu file mapping để phục vụ truy xuất ở Phase 5
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ HOÀN TẤT! Đã index {index.ntotal} vectors.")
    print(f"💾 Dữ liệu lưu tại: {index_dir}")
    return faiss_path, mapping_path

if __name__ == "__main__":
    pass