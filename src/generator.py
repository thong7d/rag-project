import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import DATA_DIR

class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang tải {model_name} lên {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def generate_answer(self, question, retrieved_passages):
        """
        Sinh câu trả lời dựa trên context và question.
        """
        # Gộp các retrieved_passages thành một chuỗi context
        context_str = "\n\n".join(retrieved_passages)
        
        # Format prompt theo đúng roadmap
        prompt = f"Answer the question using the context.\n\nContext:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer:"
        
        # Tokenize (ép giới hạn 1024 để tránh bị cắt cụt do Flan-T5 mặc định ngắn)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Sinh text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50, # Câu trả lời QA thường ngắn
                num_beams=2,       # Dùng beam search nhẹ để tăng độ chính xác
                early_stopping=True
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def cleanup(self):
        """Dọn rác VRAM sau khi dùng xong để tránh OOM."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def test_pipeline():
    """Hàm test nhanh pipeline trên 1 câu hỏi."""
    generator = RAGGenerator()
    question = "What is the capital of France?"
    contexts = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom."
    ]
    
    answer = generator.generate_answer(question, contexts)
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {answer}")
    
    generator.cleanup()

if __name__ == "__main__":
    pass