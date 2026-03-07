import os

# Tự động xác định xem code đang chạy trên Colab hay ở máy Local
IS_COLAB = os.path.exists('/content')

# Đường dẫn gốc tới dữ liệu (Trên Drive nếu là Colab, hoặc thư mục local_data nếu ở máy tính)
DATA_DIR = '/content/drive/MyDrive/rag_data' if IS_COLAB else os.path.join(os.getcwd(), 'local_data')

# Các đường dẫn con
CORPUS_DIR = os.path.join(DATA_DIR, 'corpus')
INDEX_DIR = os.path.join(DATA_DIR, 'embeddings/faiss_index')
RESULTS_DIR = os.path.join(DATA_DIR, 'experiments')

# Đảm bảo các thư mục dữ liệu này luôn tồn tại khi gọi config
for path in [CORPUS_DIR, INDEX_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)