import os
import json
import csv
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RESULTS_DIR = Path("benchmark_results")
TEST_VOICES_DIR = Path("testvoices")
MODEL_ID = "abhishtagatya/wavlm-base-960h-itw-deepfake"

def get_audio_files():
    audio_files = []
    for label in ["genuine", "spoof"]:
        folder = TEST_VOICES_DIR / label
        for file in folder.glob("*.wav"):
            audio_files.append({"path": str(file), "filename": file.name, "ground_truth": label})
    return audio_files

def analyze_audio_properties(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    max_amp = np.max(np.abs(y))
    rms = np.mean(librosa.feature.rms(y=y))
    zeros = np.sum(y == 0) / len(y)
    return duration, max_amp, rms, zeros

def main():
    print(f"Loading {MODEL_ID}...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    id2label = model.config.id2label
    print(f"Label mapping: {id2label}")
    # Based on previous results: '0': 'bona-fide', '1': 'spoof'
    genuine_idx = 0
    spoof_idx = 1
    for k, v in id2label.items():
        if "spoof" in v.lower():
            spoof_idx = int(k)
        else:
            genuine_idx = int(k)
            
    sr_expected = extractor.sampling_rate
    print(f"Expected Sample Rate: {sr_expected}")
    
    files = get_audio_files()
    results = []
    
    # TASK 1, 5: Inference
    print("\n--- TASK 1: INFERENCE ---")
    print(f"{'Filename':<12} | {'GT':<8} | {'Logit 0':<10} | {'Logit 1':<10} | {'Prob 0 (Gen)':<15} | {'Prob 1 (Spf)':<15} | {'Pred'}")
    
    for f in files:
        y, sr = librosa.load(f["path"], sr=sr_expected, mono=True)
        # Audio properties
        dur, max_a, rms, z = analyze_audio_properties(y, sr_expected)
        f["duration"] = dur
        f["max_amp"] = max_a
        f["rms"] = rms
        f["zeros"] = z
        
        inputs = extractor(y, sampling_rate=sr_expected, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits[0].numpy()
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
        
        pred_idx = np.argmax(probs)
        pred_class = "spoof" if pred_idx == spoof_idx else "genuine"
        is_correct = (pred_class == f["ground_truth"])
        
        res = {
            "filename": f["filename"],
            "ground_truth": f["ground_truth"],
            "logit_0": logits[0],
            "logit_1": logits[1],
            "probability_0": probs[0],
            "probability_1": probs[1],
            "predicted_class": pred_class,
            "correct": is_correct,
            "spoof_probability": float(probs[spoof_idx]),
            "bonafide_probability": float(probs[genuine_idx]),
            "duration": dur,
            "max_amp": float(max_a),
            "rms": float(rms)
        }
        results.append(res)
        
        print(f"{res['filename']:<12} | {res['ground_truth']:<8} | {res['logit_0']:<10.4f} | {res['logit_1']:<10.4f} | {res['probability_0']:<15.8f} | {res['probability_1']:<15.8f} | {res['predicted_class']}")

    # TASK 2: Score Distribution
    with open(RESULTS_DIR / "wavlm_scores.csv", "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["filename", "ground_truth", "spoof_probability", "bonafide_probability"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "ground_truth": r["ground_truth"],
                "spoof_probability": r["spoof_probability"],
                "bonafide_probability": r["bonafide_probability"]
            })
            
    gen_scores = [r["spoof_probability"] for r in results if r["ground_truth"] == "genuine"]
    spf_scores = [r["spoof_probability"] for r in results if r["ground_truth"] == "spoof"]
    
    dist_stats = {
        "genuine": {
            "min": float(np.min(gen_scores)),
            "max": float(np.max(gen_scores)),
            "mean": float(np.mean(gen_scores)),
            "median": float(np.median(gen_scores)),
            "std": float(np.std(gen_scores))
        },
        "spoof": {
            "min": float(np.min(spf_scores)),
            "max": float(np.max(spf_scores)),
            "mean": float(np.mean(spf_scores)),
            "median": float(np.median(spf_scores)),
            "std": float(np.std(spf_scores))
        }
    }
    
    plt.figure(figsize=(10, 6))
    plt.hist(gen_scores, bins=20, alpha=0.5, label='Genuine', color='blue', range=(0,1))
    plt.hist(spf_scores, bins=20, alpha=0.5, label='Spoof', color='red', range=(0,1))
    plt.title('WavLM Spoof Probability Distribution')
    plt.xlabel('Spoof Probability')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.savefig(RESULTS_DIR / "wavlm_score_distribution.png")
    plt.close()
    
    # TASK 3: Threshold Sweep
    thresholds = np.arange(0.05, 1.0, 0.05)
    y_true = [1 if r["ground_truth"] == "spoof" else 0 for r in results]
    y_scores = [r["spoof_probability"] for r in results]
    
    thresh_results = []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
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
            
        thresh_results.append({
            "threshold": f"{t:.2f}",
            "accuracy": f"{acc:.4f}",
            "genuine_accuracy": f"{gen_acc:.4f}",
            "spoof_accuracy": f"{spoof_acc:.4f}",
            "fpr": f"{fpr:.4f}",
            "fnr": f"{fnr:.4f}",
            "precision": f"{prec:.4f}",
            "recall": f"{rec:.4f}",
            "f1": f"{f1:.4f}"
        })
        
    with open(RESULTS_DIR / "wavlm_threshold_analysis.csv", "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=thresh_results[0].keys())
        writer.writeheader()
        writer.writerows(thresh_results)
        
    # TASK 4: Duration Test
    durations = [3, 5, 7, 10]
    duration_res = []
    
    for f in files:
        y, sr = librosa.load(f["path"], sr=sr_expected, mono=True)
        file_dur_res = {"filename": f["filename"], "ground_truth": f["ground_truth"]}
        for d in durations:
            samples = int(d * sr_expected)
            if len(y) > samples:
                y_trunc = y[:samples]
                inputs = extractor(y_trunc, sampling_rate=sr_expected, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
                file_dur_res[f"prob_{d}s"] = float(probs[spoof_idx])
            else:
                file_dur_res[f"prob_{d}s"] = None
        duration_res.append(file_dur_res)
        
    # Save overall JSON
    final_output = {
        "score_distribution": dist_stats,
        "threshold_analysis": thresh_results,
        "duration_analysis": duration_res,
        "error_analysis": {
            "false_positives": [r for r in results if r["ground_truth"] == "genuine" and r["predicted_class"] == "spoof"],
            "false_negatives": [r for r in results if r["ground_truth"] == "spoof" and r["predicted_class"] == "genuine"]
        }
    }
    
    # Cast floats recursively
    def sanitize(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
        
    with open(RESULTS_DIR / "wavlm_diagnostic.json", "w") as jf:
        json.dump(sanitize(final_output), jf, indent=2)
        
if __name__ == "__main__":
    main()
