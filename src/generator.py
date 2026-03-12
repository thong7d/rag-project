import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import DATA_DIR

class RAGGenerator:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} in 4-bit quantization on {self.device.upper()}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 4-bit Quantization Config to prevent OOM on Colab T4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
    def generate_answer(self, question, retrieved_passages):
        """
        Generate answer using strict Instruction Prompting for LLMs.
        """
        context_str = "\n\n".join(retrieved_passages)
        
        # System Prompt strictness to prevent Data Leakage and Hallucination
        prompt = (
            f"<|user|>\n"
            f"You are a highly accurate question-answering assistant.\n"
            f"Answer the following question based STRICTLY and ONLY on the provided context.\n"
            f"If the answer is not contained in the context, say 'Information not found'.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}<|end|>\n"
            f"<|assistant|>\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1, # Low temperature for factual consistency
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Strip the input prompt from the generated output
        input_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        return answer

    def cleanup(self):
        """Clean up VRAM to prevent OOM."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def test_pipeline():
    generator = RAGGenerator()
    question = "What is the capital of France?"
    contexts = ["Paris is the capital and most populous city of France."]
    
    print("\n--- TEST GENERATION ---")
    answer = generator.generate_answer(question, contexts)
    print(f"Answer: {answer}")
    print("-----------------------\n")
    generator.cleanup()

if __name__ == "__main__":
    pass