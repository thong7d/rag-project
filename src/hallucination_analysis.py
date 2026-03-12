import json
import os
import gc
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import DATA_DIR, RESULTS_DIR
from generator import RAGGenerator

def run_hallucination_analysis(k=3):
    """
    Perform hallucination analysis using an NLI Cross-Encoder model.
    This replaces rigid Exact Match with semantic entailment scoring.
    """
    test_qa_path = os.path.join(DATA_DIR, "qa", "test_qa.json")
    with open(test_qa_path, "r", encoding="utf-8") as f:
        test_qa = json.load(f)

    import random
    random.seed(42)
    eval_qa = random.sample(test_qa, min(200, len(test_qa)))

    with open(os.path.join(DATA_DIR, "processed", "passages", "chunks.json"), "r") as f:
        chunks = json.load(f)

    index = faiss.read_index(os.path.join(DATA_DIR, "embeddings", "faiss_index", "faiss.index"))

    print("1. Generating answers using RAG Generator...")
    generator = RAGGenerator()
    dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=generator.device)

    predictions_data = []
    for idx, item in enumerate(eval_qa):
        question = item["question"]
        ground_truths = item["answers"]

        q_emb = dense_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, top_indices = index.search(q_emb, k)
        retrieved_texts = [chunks[i]["text"] for i in top_indices[0]]

        pred_answer = generator.generate_answer(question, retrieved_texts)

        predictions_data.append({
            "question": question,
            "ground_truths": ground_truths,
            "contexts": retrieved_texts,
            "pred_answer": pred_answer
        })
        if (idx + 1) % 50 == 0:
            print(f"   -> Generated {idx + 1}/{len(eval_qa)} answers.")

    # Extremely critical: Free VRAM before loading the NLI model
    del index
    del dense_model
    generator.cleanup()

    print("\n2. Loading NLI Cross-Encoder for semantic entailment evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using deberta-v3-small for high accuracy but low memory footprint
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small', device=device)

    category_counts = {
        "Correct Grounded": 0,
        "Ungrounded Correct (Leakage)": 0,
        "Context Misinterpretation": 0,
        "True Hallucination": 0
    }
    analysis_log = []

    print("3. Beginning semantic hallucination analysis...")
    for idx, data in enumerate(predictions_data):
        pred = data["pred_answer"]
        if not pred.strip():
            category_counts["True Hallucination"] += 1
            continue

        # Check Semantic Correctness against Ground Truth
        is_correct = False
        for gt in data["ground_truths"]:
            # Model outputs logits for [Contradiction, Entailment, Neutral]
            scores = nli_model.predict([(gt, pred)])
            if scores[0].argmax() == 1:  # 1 represents Entailment
                is_correct = True
                break

        # Check Groundedness against Retrieved Context
        is_grounded = False
        for ctx in data["contexts"]:
            scores = nli_model.predict([(ctx, pred)])
            if scores[0].argmax() == 1:
                is_grounded = True
                break

        # Categorize the failure modes
        if is_correct and is_grounded:
            error_type = "Correct Grounded"
        elif is_correct and not is_grounded:
            error_type = "Ungrounded Correct (Leakage)"
        elif not is_correct and is_grounded:
            error_type = "Context Misinterpretation"
        else:
            error_type = "True Hallucination"

        category_counts[error_type] += 1
        analysis_log.append({
            "Question": data["question"],
            "Predicted Answer": pred,
            "Error Type": error_type
        })

        if (idx + 1) % 50 == 0:
            print(f"   -> Evaluated {idx + 1}/{len(predictions_data)} answers.")

    # Cleanup NLI model
    del nli_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    summary_df = pd.DataFrame(list(category_counts.items()), columns=["Error Type", "Frequency"])
    summary_df["Percentage (%)"] = (summary_df["Frequency"] / len(eval_qa)) * 100

    out_summary_path = os.path.join(RESULTS_DIR, "nli_hallucination_summary.csv")
    summary_df.to_csv(out_summary_path, index=False)

    print(f"\n✅ NLI Hallucination report saved to: {out_summary_path}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    pass