import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import DATA_DIR

class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def generate_answer(self, question, retrieved_passages):
        """
        Generate answer using Sandwich Prompting to prevent Mechanical Truncation and Recency Bias.
        """
        context_str = "\n\n".join(retrieved_passages)
        
        # Sandwich Prompt: Question at the top, Context in the middle, Instruction at the bottom
        prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Instruction: Based on the context provided above, briefly answer the question: {question}\n\n"
            f"Answer:"
        )
        
        # Tokenize with truncation enabled. Truncation will cut the middle context if it exceeds max_length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50, 
                num_beams=2,       
                early_stopping=True
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def cleanup(self):
        """Clean up VRAM to prevent OOM."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def test_pipeline():
    """Quick test for the generation pipeline."""
    generator = RAGGenerator()
    question = "What is the capital of France?"
    contexts = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom."
    ]
    
    print("\n--- TEST PROMPT & GENERATION ---")
    print(f"Question: {question}")
    answer = generator.generate_answer(question, contexts)
    print(f"Generated Answer: {answer}")
    print("--------------------------------\n")
    
    generator.cleanup()

if __name__ == "__main__":
    test_pipeline()