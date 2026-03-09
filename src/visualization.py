import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR

def run_visualization():
    """Đọc các file CSV kết quả và xuất biểu đồ (300 DPI) phục vụ báo cáo."""
    
    # Tạo thư mục lưu biểu đồ
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Thiết lập style học thuật
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})
    
    print("1. Đang vẽ biểu đồ Phase 5: Retrieval Performance...")
    try:
        retrieval_df = pd.read_csv(os.path.join(RESULTS_DIR, "retrieval_results.csv"))
        # Gom nhóm theo Retriever và k
        retrieval_df['Setting'] = retrieval_df['Retriever'] + " (Top-" + retrieval_df['k'].astype(str) + ")"
        
        fig, ax = plt.subplots(figsize=(8, 5))
        retrieval_melted = retrieval_df.melt(id_vars='Setting', value_vars=['Recall', 'MRR'], var_name='Metric', value_name='Score')
        sns.barplot(data=retrieval_melted, x='Setting', y='Score', hue='Metric', palette='Blues_d')
        plt.title('Retrieval Performance: BM25 vs Dense (RQ1)')
        plt.ylim(0, 1.0)
        plt.ylabel('Score')
        plt.savefig(os.path.join(plots_dir, "01_retrieval_performance.png"), dpi=300)
        plt.close()
    except FileNotFoundError:
        print("  -> Bỏ qua: Không tìm thấy retrieval_results.csv")

    print("2. Đang vẽ biểu đồ Phase 7: Generation End-to-End...")
    try:
        gen_df = pd.read_csv(os.path.join(RESULTS_DIR, "generation_results.csv"))
        gen_df['Setting'] = gen_df['Retriever'] + " (Top-" + gen_df['k'].astype(str) + ")"
        
        fig, ax = plt.subplots(figsize=(8, 5))
        gen_melted = gen_df.melt(id_vars='Setting', value_vars=['EM', 'F1'], var_name='Metric', value_name='Score')
        sns.barplot(data=gen_melted, x='Setting', y='Score', hue='Metric', palette='Set2')
        plt.title('Generation End-to-End Quality')
        plt.ylim(0, 1.0)
        plt.ylabel('Score')
        plt.savefig(os.path.join(plots_dir, "02_generation_quality.png"), dpi=300)
        plt.close()
    except FileNotFoundError:
        print("  -> Bỏ qua: Không tìm thấy generation_results.csv")

    print("3. Đang vẽ biểu đồ Phase 8: Context Compression (Trade-off)...")
    try:
        comp_df = pd.read_csv(os.path.join(RESULTS_DIR, "compression_results.csv"))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Vẽ line chart với trục X là Estimated_Tokens để phản ánh đúng khoảng cách tuyến tính
        sns.lineplot(data=comp_df, x='Estimated_Tokens', y='EM', marker='o', label='Exact Match (EM)', linewidth=2)
        sns.lineplot(data=comp_df, x='Estimated_Tokens', y='F1', marker='s', label='F1 Score', linewidth=2)
        
        plt.title('Impact of Context Size on Generation (RQ2)')
        plt.xlabel('Estimated Context Length (Tokens)')
        plt.ylabel('Score')
        plt.xticks(comp_df['Estimated_Tokens']) # Đánh dấu chính xác tại các điểm có data
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "03_context_compression.png"), dpi=300)
        plt.close()
    except FileNotFoundError:
        print("  -> Bỏ qua: Không tìm thấy compression_results.csv")

    print("4. Đang vẽ biểu đồ Phase 9: Hallucination Analysis...")
    try:
        hal_df = pd.read_csv(os.path.join(RESULTS_DIR, "hallucination_summary.csv"))
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sắp xếp để biểu đồ đẹp hơn
        hal_df = hal_df.sort_values('Percentage (%)', ascending=True)
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        
        plt.pie(hal_df['Percentage (%)'], labels=hal_df['Error Type'], autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Hallucination Distribution in Dense Top-3 (RQ3)')
        plt.axis('equal') # Giữ cho biểu đồ tròn trịa
        plt.savefig(os.path.join(plots_dir, "04_hallucination_distribution.png"), dpi=300)
        plt.close()
    except FileNotFoundError:
        print("  -> Bỏ qua: Không tìm thấy hallucination_summary.csv")

    print(f"✅ HOÀN TẤT! Tất cả biểu đồ học thuật (300 DPI) đã được lưu tại: {plots_dir}")

if __name__ == "__main__":
    pass