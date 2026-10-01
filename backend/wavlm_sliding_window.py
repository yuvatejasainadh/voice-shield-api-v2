import os
import json
import csv
import time
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

RESULTS_DIR = Path("benchmark_results")
TEST_VOICES_DIR = Path("testvoices")
MODEL_ID = "abhishtagatya/wavlm-base-960h-itw-deepfake"

WINDOW_DUR = 3.0
HOP_DUR = 0.5

def get_audio_files():
    audio_files = []
    for label in ["genuine", "spoof"]:
        folder = TEST_VOICES_DIR / label
        for file in folder.glob("*.wav"):
            audio_files.append({"path": str(file), "filename": file.name, "ground_truth": label})
    return audio_files

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return float(eer)
    
def calc_metrics(y_true, y_pred, y_scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        gen_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        spoof_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        gen_acc, spoof_acc, fpr, fnr = 0, 0, 0, 0
        
    res = {
        "Accuracy": float(acc),
        "Genuine Acc": float(gen_acc),
        "Spoof Acc": float(spoof_acc),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1)
    }
    
    if y_scores is not None and len(set(y_true)) > 1:
        try:
            res["ROC-AUC"] = float(roc_auc_score(y_true, y_scores))
            res["EER"] = float(compute_eer(y_true, y_scores))
        except:
            res["ROC-AUC"] = "N/A"
            res["EER"] = "N/A"
    else:
        res["ROC-AUC"] = "N/A"
        res["EER"] = "N/A"
        
    return res

def main():
    print(f"Loading {MODEL_ID}...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    genuine_idx = 0
    spoof_idx = 1
    sr_expected = extractor.sampling_rate
    
    files = get_audio_files()
    
    all_window_scores = []
    file_results = []
    
    window_latencies = []
    file_latencies = []
    
    # 1 & 2. Sliding Window Inference
    for f in files:
        y, sr = librosa.load(f["path"], sr=sr_expected, mono=True)
        total_samples = len(y)
        win_samples = int(WINDOW_DUR * sr_expected)
        hop_samples = int(HOP_DUR * sr_expected)
        
        # If shorter than 3s, pad with zeros
        if total_samples < win_samples:
            y = np.pad(y, (0, win_samples - total_samples), 'constant')
            total_samples = len(y)
            
        starts = range(0, total_samples - win_samples + 1, hop_samples)
        if total_samples > win_samples and (total_samples - win_samples) % hop_samples != 0:
             starts = list(starts) + [total_samples - win_samples]
        elif len(starts) == 0:
             starts = [0]
             
        file_window_probs = []
        
        t0_file = time.time()
        
        # Also compute FULL LENGTH score for comparison
        inputs_full = extractor(y, sampling_rate=sr_expected, return_tensors="pt")
        with torch.no_grad():
            out_full = model(**inputs_full)
        prob_full = float(torch.nn.functional.softmax(out_full.logits, dim=-1)[0][spoof_idx].numpy())
             
        for start_idx in starts:
            end_idx = start_idx + win_samples
            chunk = y[start_idx:end_idx]
            
            t0_win = time.time()
            inputs = extractor(chunk, sampling_rate=sr_expected, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
            t1_win = time.time()
            
            lat_ms = (t1_win - t0_win) * 1000
            window_latencies.append(lat_ms)
            
            spf_prob = float(probs[spoof_idx])
            gen_prob = float(probs[genuine_idx])
            pred_class = "spoof" if spf_prob > 0.5 else "genuine"
            
            file_window_probs.append(spf_prob)
            
            all_window_scores.append({
                "model": MODEL_ID,
                "filename": f["filename"],
                "ground_truth": f["ground_truth"],
                "window_start": start_idx / sr_expected,
                "window_end": end_idx / sr_expected,
                "spoof_probability": spf_prob,
                "bonafide_probability": gen_prob,
                "predicted_class": pred_class,
                "latency_ms": lat_ms
            })
            
        t1_file = time.time()
        file_latencies.append((t1_file - t0_file) * 1000)
        
        # 3. Aggregation Methods
        file_window_probs.sort(reverse=True)
        max_score = file_window_probs[0]
        mean_score = np.mean(file_window_probs)
        top2_mean = np.mean(file_window_probs[:2]) if len(file_window_probs) >= 2 else max_score
        
        top25_idx = max(1, int(len(file_window_probs) * 0.25))
        top25_mean = np.mean(file_window_probs[:top25_idx])
        
        spoof_votes = sum(1 for p in file_window_probs if p > 0.5)
        majority_vote = 1 if spoof_votes > (len(file_window_probs) / 2) else 0
        
        file_results.append({
            "filename": f["filename"],
            "ground_truth": f["ground_truth"],
            "full_length_score": prob_full,
            "max_window_score": float(max_score),
            "mean_window_score": float(mean_score),
            "top2_mean_score": float(top2_mean),
            "top25_mean_score": float(top25_mean),
            "majority_vote": majority_vote,
            "full_length_prediction": "spoof" if prob_full > 0.5 else "genuine",
            "max_window_prediction": "spoof" if max_score > 0.5 else "genuine",
            "num_windows": len(file_window_probs)
        })

    # Save per-window scores
    with open(RESULTS_DIR / "wavlm_sliding_window_scores.csv", "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=all_window_scores[0].keys())
        writer.writeheader()
        writer.writerows(all_window_scores)
        
    # Evaluate Aggregations
    y_true = [1 if r["ground_truth"] == "spoof" else 0 for r in file_results]
    
    aggs = {
        "FULL_LENGTH": [r["full_length_score"] for r in file_results],
        "MAX": [r["max_window_score"] for r in file_results],
        "MEAN": [r["mean_window_score"] for r in file_results],
        "TOP2_MEAN": [r["top2_mean_score"] for r in file_results],
        "TOP25_MEAN": [r["top25_mean_score"] for r in file_results],
        "MAJORITY_VOTE": [r["majority_vote"] for r in file_results]
    }
    
    agg_metrics = []
    for agg_name, scores in aggs.items():
        if agg_name == "MAJORITY_VOTE":
            y_pred = scores
            met = calc_metrics(y_true, y_pred)
        else:
            y_pred = [1 if s > 0.5 else 0 for s in scores]
            met = calc_metrics(y_true, y_pred, scores)
            
        met["Aggregation"] = agg_name
        agg_metrics.append(met)
        
    with open(RESULTS_DIR / "wavlm_sliding_window_results.csv", "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["Aggregation"] + list(agg_metrics[0].keys())[:-1])
        writer.writeheader()
        writer.writerows(agg_metrics)
        
    # 5. Threshold Analysis for MAX
    thresh_res = []
    max_scores = aggs["MAX"]
    for t in np.arange(0.05, 1.0, 0.05):
        y_pred = [1 if s >= t else 0 for s in max_scores]
        met = calc_metrics(y_true, y_pred)
        met["threshold"] = f"{t:.2f}"
        thresh_res.append(met)
        
    with open(RESULTS_DIR / "wavlm_sliding_window_thresholds.csv", "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["threshold"] + list(thresh_res[0].keys())[:-1])
        writer.writeheader()
        writer.writerows(thresh_res)
        
    # 7. Distributions
    def plot_dist(data_gen, data_spf, title, filename, ax):
        ax.hist(data_gen, bins=20, alpha=0.5, label="Genuine", range=(0,1), color='blue')
        ax.hist(data_spf, bins=20, alpha=0.5, label="Spoof", range=(0,1), color='red')
        ax.set_title(title)
        ax.legend(loc='upper right')
        
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plot_dist(
        [r["full_length_score"] for r in file_results if r["ground_truth"]=="genuine"],
        [r["full_length_score"] for r in file_results if r["ground_truth"]=="spoof"],
        "Full Length Score", "full", axs[0,0]
    )
    plot_dist(
        [r["max_window_score"] for r in file_results if r["ground_truth"]=="genuine"],
        [r["max_window_score"] for r in file_results if r["ground_truth"]=="spoof"],
        "Max Window Score", "max", axs[0,1]
    )
    plot_dist(
        [r["mean_window_score"] for r in file_results if r["ground_truth"]=="genuine"],
        [r["mean_window_score"] for r in file_results if r["ground_truth"]=="spoof"],
        "Mean Window Score", "mean", axs[1,0]
    )
    plot_dist(
        [r["top2_mean_score"] for r in file_results if r["ground_truth"]=="genuine"],
        [r["top2_mean_score"] for r in file_results if r["ground_truth"]=="spoof"],
        "Top-2 Mean Window Score", "top2", axs[1,1]
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "wavlm_sliding_window_distribution.png")
    plt.close()
    
    # 8. Latency
    avg_win_lat = np.mean(window_latencies)
    med_win_lat = np.median(window_latencies)
    p95_win_lat = np.percentile(window_latencies, 95)
    
    avg_wins_per_file = np.mean([r["num_windows"] for r in file_results])
    avg_total_lat = np.mean(file_latencies)
    
    # Estimated 10s call latency: (10s - 3s) / 0.5s + 1 = 15 windows
    est_10s_lat = 15 * avg_win_lat
    
    lat_data = {
        "avg_window_latency_ms": float(avg_win_lat),
        "median_window_latency_ms": float(med_win_lat),
        "p95_window_latency_ms": float(p95_win_lat),
        "avg_windows_per_file": float(avg_wins_per_file),
        "avg_total_latency_per_file_ms": float(avg_total_lat),
        "estimated_10s_call_latency_ms": float(est_10s_lat)
    }
    
    # JSON Output
    out_json = {
        "metrics": agg_metrics,
        "per_file": file_results,
        "latency": lat_data
    }
    with open(RESULTS_DIR / "wavlm_sliding_window.json", "w") as jf:
        json.dump(out_json, jf, indent=2)
        
if __name__ == "__main__":
    main()
