"""
Evaluate all experiments and generate comparison report
"""

from ultralytics import YOLO
import os
import pandas as pd
import json

# Configuration
EXPERIMENTS_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments"
DATASETS = {
    'rgb': 'ffb_localization.yaml',
    'depth': 'ffb_localization_depth.yaml',
    'classify': 'ffb_ripeness.yaml'
}

# Model paths (update these after training)
MODEL_PATHS = {
    'A.1 RGB Seed 42': 'runs/detect/exp_a1_rgb_seed_42/weights/best.pt',
    'A.1 RGB Seed 123': 'runs/detect/exp_a1_rgb_seed_123/weights/best.pt',
    'A.2 Depth Seed 42': 'runs/detect/exp_a2_depth_seed_42/weights/best.pt',
    'A.2 Depth Seed 123': 'runs/detect/exp_a2_depth_seed_123/weights/best.pt',
    'B.1 Cls Seed 42': 'runs/classify/exp_b1_cls_seed_42/weights/best.pt',
    'B.1 Cls Seed 123': 'runs/classify/exp_b1_cls_seed_123/weights/best.pt',
}

def evaluate_model(name, model_path, dataset_config, task='detect'):
    """Evaluate a single model"""
    
    full_model_path = os.path.join(EXPERIMENTS_DIR, model_path)
    full_data_path = os.path.join(EXPERIMENTS_DIR, dataset_config)
    
    if not os.path.exists(full_model_path):
        print(f"Model not found: {full_model_path}")
        return None
    
    print(f"Evaluating: {name}")
    
    try:
        model = YOLO(full_model_path)
        
        # Run validation
        if task == 'detect':
            results = model.val(data=full_data_path, split='test')
            return {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'Precision': results.box.mp,
                'Recall': results.box.mr,
                'Fitness': results.fitness,
            }
        else:  # classify
            results = model.val(data=full_data_path, split='test')
            return {
                'Top1_Acc': results.top1,
                'Top5_Acc': results.top5,
                'Precision': results.mp,
                'Recall': results.mr,
                'Fitness': results.fitness,
            }
            
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return None

def calculate_statistics(results_dict):
    """Calculate mean and std for experiments with multiple seeds"""
    
    grouped = {}
    
    # Group by experiment type (without seed)
    for name, metrics in results_dict.items():
        if metrics is None:
            continue
            
        # Remove seed info for grouping
        base_name = name.rsplit(' ', 1)[0]  # Remove seed number
        
        if base_name not in grouped:
            grouped[base_name] = []
        grouped[base_name].append(metrics)
    
    # Calculate stats
    stats = {}
    for exp_name, metric_list in grouped.items():
        if len(metric_list) == 0:
            continue
            
        df = pd.DataFrame(metric_list)
        mean = df.mean().to_dict()
        std = df.std().to_dict()
        
        stats[exp_name] = {
            'mean': mean,
            'std': std,
            'runs': len(metric_list)
        }
    
    return stats

def generate_report(results_dict, stats):
    """Generate comprehensive report"""
    
    report = "# LAPORAN EKSPERIMEN DETEKSI TBS\n\n"
    report += "## Hasil Per Run\n\n"
    
    # Per-run results
    report += "| Eksperimen | mAP50 | mAP50-95 | Precision | Recall | Fitness |\n"
    report += "|------------|-------|----------|-----------|--------|---------|\n"
    
    for name, metrics in results_dict.items():
        if metrics is None:
            report += f"| {name} | FAIL | FAIL | FAIL | FAIL | FAIL |\n"
        else:
            report += f"| {name} | {metrics.get('mAP50', metrics.get('Top1_Acc', 'N/A')):.4f} | "
            report += f"{metrics.get('mAP50-95', metrics.get('Top5_Acc', 'N/A')):.4f} | "
            report += f"{metrics['Precision']:.4f} | {metrics['Recall']:.4f} | "
            report += f"{metrics['Fitness']:.4f} |\n"
    
    # Statistics
    report += "\n## Statistik (Mean ± Std)\n\n"
    
    for exp_name, data in stats.items():
        report += f"### {exp_name}\n"
        report += f"- **Runs**: {data['runs']}\n"
        
        for metric, mean_val in data['mean'].items():
            std_val = data['std'][metric]
            report += f"- **{metric}**: {mean_val:.4f} ± {std_val:.4f}\n"
        
        report += "\n"
    
    # Comparison
    report += "## Perbandingan Eksperimen\n\n"
    
    # Find best mAP per experiment type
    experiment_types = {}
    for name, metrics in results_dict.items():
        if metrics is None:
            continue
        
        base_name = name.rsplit(' ', 1)[0]
        if base_name not in experiment_types:
            experiment_types[base_name] = []
        experiment_types[base_name].append((name, metrics))
    
    report += "| Tipe Eksperimen | Best Model | Best mAP50/Acc |\n"
    report += "|----------------|------------|---------------|\n"
    
    for exp_type, runs in experiment_types.items():
        # Sort by fitness
        best = max(runs, key=lambda x: x[1]['Fitness'])
        map_key = 'mAP50' if 'mAP50' in best[1] else 'Top1_Acc'
        report += f"| {exp_type} | {best[0]} | {best[1][map_key]:.4f} |\n"
    
    return report

def save_results_to_csv(results_dict, filename):
    """Save raw results to CSV"""
    
    df_data = []
    for name, metrics in results_dict.items():
        if metrics:
            row = {'Experiment': name}
            row.update(metrics)
            df_data.append(row)
    
    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    print("Mengevaluasi Semua Eksperimen")
    print("="*60)
    
    results = {}
    
    # Evaluate each model
    for name, model_path in MODEL_PATHS.items():
        # Determine task type
        task = 'classify' if 'Cls' in name else 'detect'
        
        # Determine dataset
        if 'RGB' in name:
            dataset = DATASETS['rgb']
        elif 'Depth' in name:
            dataset = DATASETS['depth']
        elif 'Cls' in name:
            dataset = DATASETS['classify']
        else:
            print(f"Unknown dataset for {name}")
            continue
        
        result = evaluate_model(name, model_path, dataset, task)
        results[name] = result
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Generate report
    report = generate_report(results, stats)
    
    # Save report
    report_path = os.path.join(EXPERIMENTS_DIR, "LAPORAN_EKSPERIMEN.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save CSV
    csv_path = os.path.join(EXPERIMENTS_DIR, "experiment_results.csv")
    save_results_to_csv(results, csv_path)
    
    print("\n" + "="*60)
    print("EVALUASI SELESAI")
    print("="*60)
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print("\nCatatan:")
    print("   - Pastikan semua model sudah dilatih sebelum evaluasi")
    print("   - Jika ada 'Model not found', jalankan training terlebih dahulu")
    print("   - Cek 'runs/' folder untuk hasil training")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
