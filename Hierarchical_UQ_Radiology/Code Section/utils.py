import matplotlib.pyplot as plt
import numpy as np

def visualize_results(hierarchy_results, threshold=0.75, save_path="uq_report_viz.png"):
    """Generates detailed UQ visualizations."""
    plt.style.use('ggplot')
    
    sentences = []
    confs = []
    for section in hierarchy_results:
        for s in section['sentences']:
            sentences.append(s['text'][:25] + "...")
            confs.append(s['confidence'])
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. Bar chart of sentence confidences
    colors = ['#27ae60' if c >= threshold else '#c0392b' for c in confs]
    ax1.barh(range(len(confs)), confs, color=colors, alpha=0.9, edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (τ={threshold})')
    ax1.set_yticks(range(len(confs)))
    ax1.set_yticklabels(sentences)
    ax1.set_xlabel("Confidence Score")
    ax1.set_title("Hierarchical Confidence Score per Sentence")
    ax1.set_xlim(0, 1.05)
    ax1.legend()

    # 2. Section aggregation
    section_names = [s['section'] for s in hierarchy_results]
    section_confs = [s['confidence'] for s in hierarchy_results]
    ax2.pie(section_confs, labels=section_names, autopct='%1.1f%%', 
            colors=['#3498db', '#9b59b6', '#f1c40f'], startangle=140, 
            explode=[0.05]*len(section_names))
    ax2.set_title("Section-Level Reliability Distribution")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    return save_path

def print_summary(hierarchy_results, threshold=0.75):
    """Prints a professional report summary to console."""
    print("="*80)
    print(f"{'SECTION':<20} | {'SENTENCE':<45} | {'CONF':<6} | {'STATUS'}")
    print("-" * 80)
    for section in hierarchy_results:
        for s in section['sentences']:
            status = "VALIDATED" if s['confidence'] >= threshold else "FLAGGED"
            print(f"{section['section']:<20} | {s['text'][:45]:<45} | {s['confidence']:.4f} | {status}")
    print("="*80)
