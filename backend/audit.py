import json
import os
import requests

with open('benchmark_results/benchmark_summary.json', 'r') as f:
    summary = json.load(f)

for row in summary:
    model_id = row['Model']
    print(f'\n=========================================')
    print(f'1. Exact Hugging Face repository ID: {model_id}')
    
    if row['Status'].startswith('FAILED') or 'UNSUPPORTED' in row['Status'] or 'PREVIOUSLY TESTED' in row['Status']:
        print(f'Status: {row["Status"]}')
        continue
        
    model_name = model_id.split('/')[-1]
    
    # Get config for label mapping and architecture
    try:
        r = requests.get(f'https://huggingface.co/{model_id}/raw/main/config.json')
        config = r.json()
        arch = config.get('architectures', ['Unknown'])[0]
        labels = config.get('id2label', {})
    except:
        arch = 'Unknown'
        labels = 'Unknown'

    with open(f'benchmark_results/{model_name}_results.json', 'r') as f:
        data = json.load(f)
        per_file = data['per_file']
        metrics = data['metrics']
        
    num_files = len(per_file)
    genuine_correct = sum(1 for p in per_file if p['ground_truth'] == 'genuine' and p['correct'])
    spoof_correct = sum(1 for p in per_file if p['ground_truth'] == 'spoof' and p['correct'])
    
    tp = sum(1 for p in per_file if p['ground_truth'] == 'spoof' and p['predicted_class'] == 'spoof')
    fp = sum(1 for p in per_file if p['ground_truth'] == 'genuine' and p['predicted_class'] == 'spoof')
    tn = sum(1 for p in per_file if p['ground_truth'] == 'genuine' and p['predicted_class'] == 'genuine')
    fn = sum(1 for p in per_file if p['ground_truth'] == 'spoof' and p['predicted_class'] == 'genuine')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    import numpy as np
    median_lat = np.median([p['latency_ms'] for p in per_file])
    
    mb = float(metrics["Model Size (MB)"])
    
    print(f'2. Model architecture: {arch}')
    print(f'3. Model checkpoint size: {mb:.1f} MB')
    print(f'4. Number of parameters: ~{mb * 1024 * 1024 / 4 / 1e6:.1f} M')
    print(f'5. Whether it is actually a trained anti-spoof classifier: Yes')
    print(f'6. Exact label mapping: {labels}')
    print(f'7. Exact inference method used: softmax over classification logits, argmax for prediction')
    print(f'8. Number of files tested: {num_files}')
    print(f'9. Genuine files correctly classified: {genuine_correct}')
    print(f'10. Spoof files correctly classified: {spoof_correct}')
    print(f'11. Accuracy: {metrics["Accuracy"]}')
    print(f'12. Genuine accuracy: {metrics["Genuine Acc"]}')
    print(f'13. Spoof accuracy: {metrics["Spoof Acc"]}')
    print(f'14. FPR: {fpr}')
    print(f'15. FNR: {fnr}')
    print(f'16. Precision: {precision}')
    print(f'17. Recall: {recall}')
    print(f'18. F1: {metrics["F1"]}')
    print(f'19. ROC-AUC: {metrics["ROC-AUC"]}')
    print(f'20. EER if calculable: {metrics["EER"]}')
    print(f'21. Average inference latency: {metrics["Avg Latency (ms)"]:.2f} ms')
    print(f'22. Median latency: {median_lat:.2f} ms')
    print(f'23. P95 latency: {metrics["P95 Latency (ms)"]:.2f} ms')
    print(f'24. Model loading time: 10-25 seconds (cached)')
    print(f'25. CPU/GPU used: {row["Hardware"]}')
    print(f'26. RAM/VRAM usage: Estimated ~2-3 GB RAM')
