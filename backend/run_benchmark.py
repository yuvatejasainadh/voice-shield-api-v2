import os
import time
import json
import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Configurations
TEST_VOICES_DIR = Path("testvoices")
BENCHMARK_RESULTS_DIR = Path("benchmark_results")
BENCHMARK_RESULTS_DIR.mkdir(exist_ok=True)

# Baseline AST results
AST_BASELINE = {
    "Model": "AST (Baseline)",
    "Status": "PREVIOUSLY TESTED — NOT RERUN",
    "Accuracy": 0.50,
    "Genuine Acc": 0.00,
    "Spoof Acc": 1.00,
    "F1": 0.6667,
    "ROC-AUC": "N/A",
    "EER": "N/A",
    "Avg Latency (ms)": "N/A",
    "P95 Latency (ms)": "N/A",
    "Model Size (MB)": "N/A",
    "Hardware": "N/A"
}

CANDIDATE_MODELS = [
    "SpeechAntiSpoofingBenchmarks/AASIST",
    "rpeddu/enhanced-rawnet2-antispoofing",
    "abhishtagatya/wavlm-base-960h-itw-deepfake",
    "garystafford/wav2vec2-deepfake-voice-detector",
    "WWWxp/wav2vec2_spoof_dection1",
    "MelodyMachine/Deepfake-audio-detection-V2"
]

def load_audio_files():
    audio_data = []
    for label, folder in [("genuine", "genuine"), ("spoof", "spoof")]:
        folder_path = TEST_VOICES_DIR / folder
        for file in folder_path.glob("*.wav"):
            try:
                # Basic validation
                duration = librosa.get_duration(path=file)
                sr = librosa.get_samplerate(file)
                audio_data.append({
                    "path": str(file),
                    "filename": file.name,
                    "ground_truth": label,
                    "duration": duration,
                    "sample_rate_original": sr
                })
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return audio_data

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def map_labels(model_id2label):
    # Determine which label index is genuine and which is spoof
    genuine_idx, spoof_idx = None, None
    for idx, label in model_id2label.items():
        l_lower = label.lower()
        if "bona" in l_lower or "real" in l_lower or "genuine" in l_lower:
            genuine_idx = int(idx)
        elif "spoof" in l_lower or "fake" in l_lower:
            spoof_idx = int(idx)
            
    if genuine_idx is None and spoof_idx is not None:
        genuine_idx = 1 - spoof_idx
    elif spoof_idx is None and genuine_idx is not None:
        spoof_idx = 1 - genuine_idx
        
    return genuine_idx, spoof_idx

def run_benchmark():
    audio_files = load_audio_files()
    avg_duration = np.mean([f["duration"] for f in audio_files])
    print(f"Loaded {len(audio_files)} files. Avg duration: {avg_duration:.2f}s")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware_info = f"Device: {device}"
    if device == "cuda":
        hardware_info += f", GPU: {torch.cuda.get_device_name(0)}"
    
    summary_results = [AST_BASELINE]
    
    for model_id in CANDIDATE_MODELS:
        print(f"\n--- Benchmarking {model_id} ---")
        model_name = model_id.split("/")[-1]
        
        result = {
            "Model": model_id,
            "Status": "FAILED",
            "Accuracy": "N/A",
            "Genuine Acc": "N/A",
            "Spoof Acc": "N/A",
            "F1": "N/A",
            "ROC-AUC": "N/A",
            "EER": "N/A",
            "Avg Latency (ms)": "N/A",
            "P95 Latency (ms)": "N/A",
            "Model Size (MB)": "N/A",
            "Hardware": hardware_info
        }
        
        try:
            # Measure load time
            t0 = time.time()
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True).to(device)
            except Exception as e:
                print(f"Failed to load model {model_id}: {e}")
                if "config.json" in str(e) or "404" in str(e) or "invalid" in str(e).lower() or "not found" in str(e).lower():
                    result["Status"] = "FAILED (Invalid/No Config)"
                elif "memory" in str(e).lower() or "allocate" in str(e).lower():
                    result["Status"] = "RESOURCE_LIMITATION"
                else:
                    result["Status"] = f"FAILED ({type(e).__name__})"
                summary_results.append(result)
                continue
                
            model.eval()
            load_time = time.time() - t0
            print(f"Loaded in {load_time:.2f}s")
            
            model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
            result["Model Size (MB)"] = f"{model_size:.1f}"
            
            genuine_idx, spoof_idx = map_labels(model.config.id2label)
            if genuine_idx is None or spoof_idx is None:
                print(f"Could not map labels for {model_id}. id2label: {model.config.id2label}")
                result["Status"] = "FAILED (Invalid Labels)"
                summary_results.append(result)
                continue
                
            target_sr = feature_extractor.sampling_rate
            
            per_file_results = []
            latencies = []
            y_true = []
            y_scores = []
            y_pred = []
            
            # Warm up
            dummy_input = torch.zeros(1, target_sr).to(device)
            if hasattr(feature_extractor, "return_attention_mask") and feature_extractor.return_attention_mask:
                dummy_inputs = feature_extractor(dummy_input.cpu().numpy()[0], sampling_rate=target_sr, return_tensors="pt")
            else:
                dummy_inputs = feature_extractor(dummy_input.cpu().numpy()[0], sampling_rate=target_sr, return_tensors="pt")
            dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
            with torch.no_grad():
                model(**dummy_inputs)
            
            for file_info in audio_files:
                waveform, sr = librosa.load(file_info["path"], sr=target_sr)
                
                inputs = feature_extractor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                t_inf_start = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                t_inf_end = time.time()
                
                latency_ms = (t_inf_end - t_inf_start) * 1000
                latencies.append(latency_ms)
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                spoof_prob = float(probs[spoof_idx])
                bonafide_prob = float(probs[genuine_idx])
                pred_class = "spoof" if spoof_prob > bonafide_prob else "genuine"
                confidence = max(spoof_prob, bonafide_prob)
                
                is_correct = pred_class == file_info["ground_truth"]
                
                per_file_results.append({
                    "model": model_id,
                    "filename": file_info["filename"],
                    "ground_truth": file_info["ground_truth"],
                    "predicted_class": pred_class,
                    "correct": is_correct,
                    "spoof_probability": spoof_prob,
                    "bonafide_probability": bonafide_prob,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "sample_rate": target_sr,
                    "duration_seconds": file_info["duration"]
                })
                
                y_true.append(1 if file_info["ground_truth"] == "spoof" else 0)
                y_scores.append(spoof_prob)
                y_pred.append(1 if pred_class == "spoof" else 0)
                
            # Compute metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_true, y_scores)
            except:
                auc = "N/A"
                
            eer = compute_eer(y_true, y_scores) if len(set(y_true)) > 1 else "N/A"
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                gen_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
                spoof_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                gen_acc, spoof_acc = 0, 0
                
            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            
            result.update({
                "Status": "PASS",
                "Accuracy": acc,
                "Genuine Acc": gen_acc,
                "Spoof Acc": spoof_acc,
                "F1": f1,
                "ROC-AUC": auc,
                "EER": eer,
                "Avg Latency (ms)": avg_lat,
                "P95 Latency (ms)": p95_lat
            })
            
            summary_results.append(result)
            
            # Save per-model CSV
            csv_path = BENCHMARK_RESULTS_DIR / f"{model_name}_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_file_results[0].keys())
                writer.writeheader()
                writer.writerows(per_file_results)
                
            # Save per-model JSON
            json_path = BENCHMARK_RESULTS_DIR / f"{model_name}_results.json"
            with open(json_path, "w") as f:
                json.dump({
                    "model": model_id,
                    "metrics": {k: v for k, v in result.items() if k not in ["Model", "Status"]},
                    "per_file": per_file_results
                }, f, indent=2)
                
            # Save per-model MD
            md_path = BENCHMARK_RESULTS_DIR / f"{model_name}_report.md"
            with open(md_path, "w") as f:
                f.write(f"# Benchmark Report: {model_id}\n\n")
                f.write(f"**Status**: {result['Status']}\n")
                f.write(f"**Accuracy**: {result['Accuracy']}\n")
                f.write(f"**Genuine Acc**: {result['Genuine Acc']}\n")
                f.write(f"**Spoof Acc**: {result['Spoof Acc']}\n")
                f.write(f"**F1 Score**: {result['F1']}\n")
                f.write(f"**Average Latency**: {result['Avg Latency (ms)']} ms\n")
                
        except Exception as e:
            print(f"Failed during inference for {model_id}: {e}")
            result["Status"] = f"FAILED ({type(e).__name__})"
            summary_results.append(result)
            
    # Save Summary
    with open(BENCHMARK_RESULTS_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
        
    with open(BENCHMARK_RESULTS_DIR / "benchmark_results.csv", "w", newline="") as f:
        keys = summary_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_results)
        
if __name__ == '__main__':
    run_benchmark()
