import json
import os
import nltk
from transformers import AutoTokenizer

# Tải bộ dữ liệu tách câu của NLTK (ẩn log)
import logging
logging.getLogger("nltk").setLevel(logging.ERROR)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from config import CORPUS_DIR, DATA_DIR

def semantic_chunking(text, max_tokens, overlap_tokens, tokenizer):
    """
    Cắt chunk dựa trên ranh giới câu (Semantic Boundaries) để bảo toàn ngữ nghĩa.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Mã hóa câu để đếm token chính xác theo all-MiniLM-L6-v2
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        
        # Nếu chỉ 1 câu đã vượt max_tokens, đành phải cắt cơ học câu đó
        if sentence_length > max_tokens:
            if current_chunk:
                chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))
                current_chunk = []
                current_length = 0
            
            # Cắt cơ học câu quá dài
            i = 0
            while i < sentence_length:
                chunk_slice = sentence_tokens[i : i + max_tokens]
                chunks.append(tokenizer.decode(chunk_slice, skip_special_tokens=True))
                i += (max_tokens - overlap_tokens)
            continue
            
        # Nếu thêm câu này vào chunk hiện tại mà vượt ngưỡng -> Lưu chunk cũ
        if current_length + sentence_length > max_tokens:
            chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))
            # Logic Overlap cho Semantic: Giữ lại câu cuối cùng của chunk trước
            if len(current_chunk) > 0:
                # Ước lượng overlap bằng cách lấy câu cuối
                last_sentence_tokens = tokenizer.encode(sentences[sentences.index(sentence)-1], add_special_tokens=False)
                current_chunk = last_sentence_tokens + sentence_tokens
                current_length = len(current_chunk)
            else:
                current_chunk = sentence_tokens
                current_length = sentence_length
        else:
            current_chunk.extend(sentence_tokens)
            current_length += sentence_length
            
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))
        
    return chunks

def process_chunking(chunk_size=200, overlap=50, input_filename="passages.json", output_filename="chunks.json"):
    input_filepath = os.path.join(CORPUS_DIR, input_filename)
    output_dir = os.path.join(DATA_DIR, 'processed', 'passages')
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Nếu file đã tồn tại, tự động ghi đè để cập nhật thuật toán mới (Force update)
    print(f"1. Đang tải dữ liệu gốc từ {input_filepath}...")
    with open(input_filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print("2. Đang khởi tạo Tokenizer (all-MiniLM-L6-v2) và NLTK Sentence Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    chunks = []
    print(f"3. Bắt đầu Semantic Chunking (Size: {chunk_size}, Overlap: ~{overlap})...")
    
    for idx, doc in enumerate(corpus):
        semantic_texts = semantic_chunking(doc["text"], chunk_size, overlap, tokenizer)
        
        for chunk_idx, chunk_text in enumerate(semantic_texts):
            if len(chunk_text.strip()) > 10: # Lọc các chunk quá ngắn chứa rác
                chunks.append({
                    "chunk_id": f"{doc['id']}_{chunk_idx}",
                    "parent_id": doc['id'],
                    "title": doc['title'],
                    "text": chunk_text
                })
            
        if (idx + 1) % 10000 == 0:
            print(f"   -> Đã xử lý {idx + 1}/{len(corpus)} documents, tạo ra {len(chunks)} semantic chunks...")

    print("4. Đang ghi file chunks xuống Google Drive...")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ HOÀN TẤT! Từ {len(corpus)} docs ban đầu đã sinh ra {len(chunks)} semantic chunks.")
    print(f"💾 File lưu tại: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    pass