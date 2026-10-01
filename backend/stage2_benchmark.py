"""
Stage 2: Full Benchmark Engine
- Loads deduplicated 2000-file dataset (1000 genuine + 1000 spoof)
- Stratified 20/20/60 split (dev / val / test)
- Tests each model with FULL_LENGTH and sliding-window strategies
- WavLM gets 9 aggregation methods
- Threshold sweep on dev set only
- All metrics, latencies, confusion matrices, ROC curves
- Saves all CSVs, JSONs, PNGs, and Markdown reports
"""
import gc
import json
import csv
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("benchmark_results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Candidate models (pre-verified in earlier benchmark) ─────────────────────
CANDIDATE_MODELS = [
    {
        "id": "abhishtagatya/wavlm-base-960h-itw-deepfake",
        "name": "wavlm",
        "genuine_idx": 0,
        "spoof_idx": 1,
    },
    {
        "id": "MelodyMachine/Deepfake-audio-detection-V2",
        "name": "melodymachine",
        "genuine_idx": 1,   # label 0='fake', 1='real'
        "spoof_idx": 0,
    },
    {
        "id": "WWWxp/wav2vec2_spoof_dection1",
        "name": "wwwxp_w2v2",
        "genuine_idx": 0,
        "spoof_idx": 1,
    },
]

WINDOW_DUR = 3.0
HOP_DUR    = 0.5

# ── Dataset loading ───────────────────────────────────────────────────────────
def load_dataset():
    """Load deduplicated file list."""
    entries = []
    COPY_SUFFIX = " - Copy"
    for label in ["genuine", "spoof"]:
        folder = Path("testvoices") / label
        for p in sorted(folder.glob("*.wav")):
            if COPY_SUFFIX in p.stem:
                continue   # skip Windows duplicates
            entries.append({"path": str(p), "filename": p.name, "label": label})
    return entries

def stratified_split(entries, dev_frac=0.20, val_frac=0.20, seed=42):
    """Stratified 20/20/60 split."""
    gen = [e for e in entries if e["label"] == "genuine"]
    spf = [e for e in entries if e["label"] == "spoof"]

    def split_class(items):
        # first cut: dev + remaining
        dev, rest = train_test_split(items, test_size=1-dev_frac, random_state=seed)
        # second cut from rest: val vs test
        val_frac_of_rest = val_frac / (1 - dev_frac)
        val, test = train_test_split(rest, test_size=1-val_frac_of_rest, random_state=seed)
        return dev, val, test

    g_dev, g_val, g_test = split_class(gen)
    s_dev, s_val, s_test = split_class(spf)

    dev  = g_dev  + s_dev
    val  = g_val  + s_val
    test = g_test + s_test
    return dev, val, test

# ── Metrics helpers ───────────────────────────────────────────────────────────
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])

def fpr_at_recall(y_true, y_score, target_recall):
    """FPR at a given spoof-recall level."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score, pos_label=1)
    for fpr_val, tpr_val in zip(fpr_arr, tpr_arr):
        if tpr_val >= target_recall:
            return float(fpr_val)
    return float(fpr_arr[-1])

def full_metrics(y_true, y_pred, y_score=None):
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    gen_acc   = tn/(tn+fp) if (tn+fp)>0 else 0.0
    spoof_acc = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr_val   = fp/(fp+tn) if (fp+tn)>0 else 0.0
    fnr_val   = fn/(fn+tp) if (fn+tp)>0 else 0.0
    m = {
        "accuracy": float(acc), "genuine_acc": float(gen_acc),
        "spoof_acc": float(spoof_acc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "fpr": float(fpr_val), "fnr": float(fnr_val),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if y_score is not None and len(set(y_true))>1:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
            m["eer"] = compute_eer(y_true, y_score)
            for r in [0.50, 0.70, 0.80, 0.90]:
                m[f"fpr_at_recall_{int(r*100)}"] = fpr_at_recall(y_true, y_score, r)
        except Exception:
            m["roc_auc"] = None; m["eer"] = None
    return m

def score_dist_stats(scores):
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "p5":  float(np.percentile(arr, 5)),  "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def plot_roc(y_true, y_score, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def plot_score_dist(gen_scores, spf_scores, title, path):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(gen_scores, bins=40, alpha=0.55, label="Genuine", range=(0,1), color="steelblue")
    ax.hist(spf_scores, bins=40, alpha=0.55, label="Spoof",   range=(0,1), color="firebrick")
    ax.set_title(title); ax.set_xlabel("Spoof probability"); ax.set_ylabel("Count")
    ax.legend(); plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

# ── Full-length inference ─────────────────────────────────────────────────────
def run_full_length(model_cfg, extractor, model, files, split_name):
    sid = model_cfg["name"]
    spoof_idx   = model_cfg["spoof_idx"]
    genuine_idx = model_cfg["genuine_idx"]
    sr = extractor.sampling_rate

    rows = []
    latencies = []
    for f in files:
        y, _ = librosa.load(f["path"], sr=sr, mono=True)
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        t0 = time.time()
        with torch.no_grad():
            logits = model(**inputs).logits
        lat = (time.time()-t0)*1000
        latencies.append(lat)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        spoof_prob  = float(probs[spoof_idx])
        genuine_prob= float(probs[genuine_idx])
        pred = "spoof" if spoof_prob > 0.5 else "genuine"
        rows.append({
            "filename": f["filename"],
            "ground_truth": f["label"],
            "prediction": pred,
            "spoof_probability": spoof_prob,
            "genuine_probability": genuine_prob,
            "raw_score": spoof_prob,
            "threshold": 0.5,
            "correct": pred == f["label"],
            "duration": None,
            "inference_latency_ms": lat,
            "split": split_name,
        })
    return rows, latencies

# ── Sliding-window inference ──────────────────────────────────────────────────
def run_sliding_window(model_cfg, extractor, model, files, split_name,
                       window_sec=3.0, hop_sec=0.5):
    sid = model_cfg["name"]
    spoof_idx   = model_cfg["spoof_idx"]
    genuine_idx = model_cfg["genuine_idx"]
    sr = extractor.sampling_rate
    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    file_rows   = []
    window_rows = []
    window_lats = []

    for f in files:
        y, _ = librosa.load(f["path"], sr=sr, mono=True)
        # Pad if shorter than window
        if len(y) < win_samples:
            y = np.pad(y, (0, win_samples - len(y)), "constant")

        total = len(y)
        starts = list(range(0, total - win_samples + 1, hop_samples))
        if not starts:
            starts = [0]

        win_probs = []
        for s in starts:
            chunk = y[s:s+win_samples]
            inp = extractor(chunk, sampling_rate=sr, return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                logits = model(**inp).logits
            lat = (time.time()-t0)*1000
            window_lats.append(lat)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            sp = float(probs[spoof_idx])
            gp = float(probs[genuine_idx])
            pred_w = "spoof" if sp > 0.5 else "genuine"
            win_probs.append(sp)
            window_rows.append({
                "filename": f["filename"],
                "ground_truth": f["label"],
                "window_start": s/sr,
                "window_end":  (s+win_samples)/sr,
                "spoof_probability": sp,
                "bonafide_probability": gp,
                "predicted_class": pred_w,
                "latency_ms": lat,
                "split": split_name,
            })

        win_probs_sorted = sorted(win_probs, reverse=True)
        n = len(win_probs_sorted)
        top2   = float(np.mean(win_probs_sorted[:2])) if n>=2 else win_probs_sorted[0]
        top5   = float(np.mean(win_probs_sorted[:5])) if n>=5 else float(np.mean(win_probs_sorted))
        top10  = float(np.mean(win_probs_sorted[:10])) if n>=10 else float(np.mean(win_probs_sorted))
        top25p = float(np.mean(win_probs_sorted[:max(1,int(n*0.25))]))
        top10p = float(np.mean(win_probs_sorted[:max(1,int(n*0.10))]))
        majority_spoof = sum(1 for p in win_probs if p>0.5) > n/2

        file_rows.append({
            "filename": f["filename"],
            "ground_truth": f["label"],
            "max_score":   float(win_probs_sorted[0]),
            "mean_score":  float(np.mean(win_probs)),
            "top2_mean":   top2,
            "top5_mean":   top5,
            "top10_mean":  top10,
            "top25pct":    top25p,
            "top10pct":    top10p,
            "majority_vote": int(majority_spoof),
            "num_windows": n,
            "split": split_name,
        })

    return file_rows, window_rows, window_lats

# ── Aggregation evaluation ────────────────────────────────────────────────────
def eval_aggregation(file_rows, agg_key, threshold=0.5):
    y_true  = [1 if r["ground_truth"]=="spoof" else 0 for r in file_rows]
    if agg_key == "majority_vote":
        y_score = [float(r["majority_vote"]) for r in file_rows]
        y_pred  = [int(r["majority_vote"]) for r in file_rows]
    else:
        y_score = [r[agg_key] for r in file_rows]
        y_pred  = [1 if s >= threshold else 0 for s in y_score]
    return full_metrics(y_true, y_pred, y_score if agg_key!="majority_vote" else None)

# ── Threshold sweep ───────────────────────────────────────────────────────────
def threshold_sweep(scores_and_labels, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    y_true   = [1 if lab=="spoof" else 0 for _, lab in scores_and_labels]
    y_scores = [sc for sc, _ in scores_and_labels]
    rows = []
    for t in thresholds:
        y_pred = [1 if s>=t else 0 for s in y_scores]
        m = full_metrics(y_true, y_pred, y_scores)
        m["threshold"] = float(round(t, 2))
        rows.append(m)
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    entries = load_dataset()
    print(f"  Total after dedup: {len(entries)}  "
          f"({sum(1 for e in entries if e['label']=='genuine')} genuine, "
          f"{sum(1 for e in entries if e['label']=='spoof')} spoof)")

    dev, val, test = stratified_split(entries)
    print(f"  Splits: dev={len(dev)}, val={len(val)}, test={len(test)}")
    split_map = {e["filename"]: "dev"  for e in dev}
    split_map.update({e["filename"]: "val"  for e in val})
    split_map.update({e["filename"]: "test" for e in test})

    # Save split assignments
    split_records = [{"filename": e["filename"], "label": e["label"],
                      "split": split_map[e["filename"]]} for e in entries]
    with open(RESULTS_DIR/"split_assignments.json","w") as f:
        json.dump(split_records, f, indent=2)

    summary_rows = []
    all_results  = {}

    for mcfg in CANDIDATE_MODELS:
        mid  = mcfg["id"]
        name = mcfg["name"]
        print(f"\n{'='*60}")
        print(f"MODEL: {mid}")

        # Load model
        try:
            t_load = time.time()
            extractor = AutoFeatureExtractor.from_pretrained(mid)
            model = AutoModelForAudioClassification.from_pretrained(mid)
            model.eval()
            load_time = time.time() - t_load
            param_count = sum(p.numel() for p in model.parameters())
            size_mb = param_count * 4 / (1024**2)
            print(f"  Loaded in {load_time:.1f}s  params={param_count/1e6:.1f}M  "
                  f"size≈{size_mb:.0f}MB")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            summary_rows.append({
                "model": mid, "strategy": "N/A", "status": f"LOAD_FAILED: {e}",
                "accuracy": None, "genuine_acc": None, "spoof_acc": None,
                "f1": None, "roc_auc": None, "eer": None,
                "fpr": None, "fnr": None,
                "avg_latency_ms": None, "size_mb": None
            })
            continue

        model_results = {}

        # ── FULL_LENGTH on all splits ─────────────────────────────────────
        print("  Running FULL_LENGTH inference...")
        all_fl_rows = []
        for split_name, split_files in [("dev", dev), ("val", val), ("test", test)]:
            fl_rows, fl_lats = run_full_length(mcfg, extractor, model,
                                               split_files, split_name)
            all_fl_rows.extend(fl_rows)

        # Save per-file CSV
        fl_csv = RESULTS_DIR / f"{name}_full_length_results.csv"
        if all_fl_rows:
            with open(fl_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_fl_rows[0].keys())
                w.writeheader(); w.writerows(all_fl_rows)

        # Evaluate each split
        for split_name in ["dev", "val", "test"]:
            rows_s = [r for r in all_fl_rows if r["split"]==split_name]
            yt = [1 if r["ground_truth"]=="spoof" else 0 for r in rows_s]
            yp = [1 if r["prediction"]=="spoof"   else 0 for r in rows_s]
            ys = [r["spoof_probability"] for r in rows_s]
            m  = full_metrics(yt, yp, ys)
            lats = [r["inference_latency_ms"] for r in rows_s]
            m["avg_lat"] = float(np.mean(lats)) if lats else None
            m["p95_lat"] = float(np.percentile(lats,95)) if lats else None
            model_results[f"full_length_{split_name}"] = m
            if split_name=="val":
                print(f"    FULL_LENGTH val: acc={m['accuracy']:.3f}  "
                      f"spoof_acc={m['spoof_acc']:.3f}  f1={m['f1']:.3f}  "
                      f"roc_auc={m.get('roc_auc','N/A')}")

        # Score distribution plot (test set, full length)
        test_fl = [r for r in all_fl_rows if r["split"]=="test"]
        gen_sc = [r["spoof_probability"] for r in test_fl if r["ground_truth"]=="genuine"]
        spf_sc = [r["spoof_probability"] for r in test_fl if r["ground_truth"]=="spoof"]
        plot_score_dist(gen_sc, spf_sc, f"{name} FULL_LENGTH score dist (test)",
                        PLOTS_DIR/f"{name}_full_length_dist.png")

        # ROC on test set
        test_yt = [1 if r["ground_truth"]=="spoof" else 0 for r in test_fl]
        test_ys = [r["spoof_probability"] for r in test_fl]
        if len(set(test_yt))>1:
            plot_roc(test_yt, test_ys, f"{name} FULL_LENGTH ROC (test)",
                     PLOTS_DIR/f"{name}_full_length_roc.png")

        # Confusion matrix (test set)
        test_yp = [1 if r["prediction"]=="spoof" else 0 for r in test_fl]
        cm = confusion_matrix(test_yt, test_yp, labels=[0,1])
        plot_confusion_matrix(cm, ["Genuine","Spoof"],
                              f"{name} FULL_LENGTH (test)",
                              PLOTS_DIR/f"{name}_full_length_cm.png")

        # ── WavLM: also do sliding window ────────────────────────────────
        if name == "wavlm":
            print("  Running SLIDING-WINDOW inference (WavLM only)...")
            all_sw_file_rows = []
            all_sw_win_rows  = []
            all_sw_win_lats  = []

            for split_name, split_files in [("dev", dev), ("val", val), ("test", test)]:
                print(f"    Split: {split_name} ({len(split_files)} files)...")
                fr, wr, wl = run_sliding_window(mcfg, extractor, model,
                                                 split_files, split_name)
                all_sw_file_rows.extend(fr)
                all_sw_win_rows.extend(wr)
                all_sw_win_lats.extend(wl)

            # Save window-level CSV
            sw_csv = RESULTS_DIR / f"{name}_sliding_window_wins.csv"
            if all_sw_win_rows:
                with open(sw_csv,"w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=all_sw_win_rows[0].keys())
                    w.writeheader(); w.writerows(all_sw_win_rows)

            # Save file-level CSV
            sw_file_csv = RESULTS_DIR / f"{name}_sliding_window_files.csv"
            if all_sw_file_rows:
                with open(sw_file_csv,"w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=all_sw_file_rows[0].keys())
                    w.writeheader(); w.writerows(all_sw_file_rows)

            AGG_KEYS = ["max_score","mean_score","top2_mean","top5_mean",
                        "top10_mean","top25pct","top10pct","majority_vote"]

            # Threshold selection on DEV set using MAX score
            dev_sw = [r for r in all_sw_file_rows if r["split"]=="dev"]
            dev_scores_labels = [(r["max_score"], r["ground_truth"]) for r in dev_sw]
            dev_thresh_rows = threshold_sweep(dev_scores_labels)
            best_thresh_row = max(dev_thresh_rows, key=lambda x: x["f1"])
            best_thresh = float(best_thresh_row["threshold"])
            print(f"    Dev best threshold (max_score): {best_thresh:.2f}  "
                  f"f1={best_thresh_row['f1']:.3f}")

            # Save dev threshold sweep
            thresh_csv = RESULTS_DIR / f"{name}_sw_threshold_sweep.csv"
            with open(thresh_csv,"w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=dev_thresh_rows[0].keys())
                w.writeheader(); w.writerows(dev_thresh_rows)

            # Evaluate each aggregation on val and test
            for split_name in ["dev","val","test"]:
                rows_s = [r for r in all_sw_file_rows if r["split"]==split_name]
                for agg in AGG_KEYS:
                    # Use best_thresh for score-based aggs, majority vote for that
                    t = best_thresh if agg != "majority_vote" else 0.5
                    m = eval_aggregation(rows_s, agg, threshold=t)
                    model_results[f"sw_{agg}_{split_name}"] = m
                    if split_name=="val":
                        print(f"    SW {agg:<14} val: "
                              f"acc={m['accuracy']:.3f}  "
                              f"spoof_acc={m['spoof_acc']:.3f}  "
                              f"f1={m['f1']:.3f}  "
                              f"roc_auc={m.get('roc_auc','N/A')}")

            # Window latency stats
            avg_wl = np.mean(all_sw_win_lats)
            med_wl = np.median(all_sw_win_lats)
            p95_wl = np.percentile(all_sw_win_lats, 95)
            avg_nw = np.mean([r["num_windows"] for r in all_sw_file_rows])
            # Estimated 10s latency: (10-3)/0.5+1 = 15 windows
            est_10s = 15 * avg_wl
            model_results["window_latency"] = {
                "avg_ms": float(avg_wl), "median_ms": float(med_wl),
                "p95_ms": float(p95_wl), "avg_windows_per_file": float(avg_nw),
                "estimated_10s_call_ms": float(est_10s),
            }
            print(f"    Window latency avg={avg_wl:.1f}ms  "
                  f"p95={p95_wl:.1f}ms  est_10s={est_10s:.0f}ms")

            # Score distribution plots for best agg
            best_agg = max(
                [(a, model_results.get(f"sw_{a}_val",{})) for a in AGG_KEYS
                 if a != "majority_vote"],
                key=lambda x: x[1].get("f1", 0)
            )[0]
            test_sw = [r for r in all_sw_file_rows if r["split"]=="test"]
            gen_sw_sc = [r[best_agg] for r in test_sw if r["ground_truth"]=="genuine"]
            spf_sw_sc = [r[best_agg] for r in test_sw if r["ground_truth"]=="spoof"]
            plot_score_dist(gen_sw_sc, spf_sw_sc,
                            f"WavLM SW {best_agg} score dist (test)",
                            PLOTS_DIR/f"wavlm_sw_{best_agg}_dist.png")

            # ROC and CM for best agg on test
            test_yt_sw = [1 if r["ground_truth"]=="spoof" else 0 for r in test_sw]
            test_ys_sw = [r[best_agg] for r in test_sw]
            test_yp_sw = [1 if s>=best_thresh else 0 for s in test_ys_sw]
            if len(set(test_yt_sw))>1:
                plot_roc(test_yt_sw, test_ys_sw,
                         f"WavLM SW {best_agg} ROC (test)",
                         PLOTS_DIR/f"wavlm_sw_{best_agg}_roc.png")
            cm_sw = confusion_matrix(test_yt_sw, test_yp_sw, labels=[0,1])
            plot_confusion_matrix(cm_sw, ["Genuine","Spoof"],
                                  f"WavLM SW {best_agg} (test)",
                                  PLOTS_DIR/f"wavlm_sw_{best_agg}_cm.png")

            # Robustness by duration bucket on test set
            print("    Robustness by duration bucket (test)...")
            import soundfile as sf
            buckets = [("<3s", 0,3), ("3-5s",3,5), ("5-7s",5,7),
                       ("7-10s",7,10), (">10s",10,9999)]
            for bname, bmin, bmax in buckets:
                bucket_rows = []
                for r in test_sw:
                    try:
                        dur = sf.info(next(e["path"] for e in entries
                                          if e["filename"]==r["filename"])).duration
                    except:
                        continue
                    if bmin <= dur < bmax:
                        bucket_rows.append(r)
                if len(bucket_rows) >= 10:
                    byt = [1 if r["ground_truth"]=="spoof" else 0 for r in bucket_rows]
                    bys = [r[best_agg] for r in bucket_rows]
                    byp = [1 if s>=best_thresh else 0 for s in bys]
                    bm = full_metrics(byt, byp, bys)
                    model_results[f"sw_{best_agg}_test_bucket_{bname}"] = bm
                    print(f"      {bname}: n={len(bucket_rows)}  "
                          f"spoof_acc={bm['spoof_acc']:.3f}  "
                          f"fpr={bm['fpr']:.3f}")

        # ── Add to summary ────────────────────────────────────────────────
        val_key = f"sw_max_score_val" if name=="wavlm" else "full_length_val"
        test_key= f"sw_max_score_test" if name=="wavlm" else "full_length_test"
        vm = model_results.get(val_key, {})
        tm = model_results.get(test_key, {})

        fl_lats_test = ([r["inference_latency_ms"]
                         for r in all_fl_rows if r["split"]=="test"]
                        if 'all_fl_rows' in dir() else [])

        summary_rows.append({
            "model": mid, "strategy": "FULL_LENGTH",
            "status": "PASS",
            "accuracy":    tm.get("accuracy"),
            "genuine_acc": tm.get("genuine_acc"),
            "spoof_acc":   tm.get("spoof_acc"),
            "f1":          tm.get("f1"),
            "roc_auc":     tm.get("roc_auc"),
            "eer":         tm.get("eer"),
            "fpr":         tm.get("fpr"),
            "fnr":         tm.get("fnr"),
            "avg_latency_ms": (np.mean(fl_lats_test)
                               if fl_lats_test else None),
            "size_mb": round(size_mb, 1),
        })

        if name == "wavlm":
            best_agg_val = max(
                [(a, model_results.get(f"sw_{a}_val", {})) for a in AGG_KEYS
                 if a != "majority_vote"],
                key=lambda x: x[1].get("f1", 0)
            )
            ba_name, ba_val = best_agg_val
            ba_test = model_results.get(f"sw_{ba_name}_test", {})
            summary_rows.append({
                "model": mid, "strategy": f"SW_{ba_name.upper()}",
                "status": "PASS",
                "accuracy":    ba_test.get("accuracy"),
                "genuine_acc": ba_test.get("genuine_acc"),
                "spoof_acc":   ba_test.get("spoof_acc"),
                "f1":          ba_test.get("f1"),
                "roc_auc":     ba_test.get("roc_auc"),
                "eer":         ba_test.get("eer"),
                "fpr":         ba_test.get("fpr"),
                "fnr":         ba_test.get("fnr"),
                "avg_latency_ms": model_results.get("window_latency",{}).get("avg_ms"),
                "size_mb": round(size_mb, 1),
            })

        all_results[name] = model_results

        # ── Free memory ───────────────────────────────────────────────────
        del model, extractor
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  Memory released.")

    # ── Save summary CSV ─────────────────────────────────────────────────────
    with open(RESULTS_DIR/"benchmark_results.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    # ── Save full JSON ───────────────────────────────────────────────────────
    def jdefault(o):
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32,  np.int64)):    return int(o)
        return str(o)

    with open(RESULTS_DIR/"benchmark_summary.json","w") as f:
        json.dump({"summary": summary_rows, "detail": all_results},
                  f, indent=2, default=jdefault)

    # ── Print final table ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"FINAL COMPARISON TABLE (test set metrics)")
    print(f"{'='*100}")
    hdr = f"{'Model':<45} {'Strategy':<18} {'Acc':>5} {'GenAcc':>7} {'SpfAcc':>7} {'F1':>6} {'AUC':>6} {'EER':>5} {'FPR':>5} {'Lat(ms)':>8} {'MB':>6}"
    print(hdr)
    print("-"*len(hdr))
    for r in summary_rows:
        def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{r['model']:<45} {r['strategy']:<18} "
              f"{fmt(r['accuracy']):>5} {fmt(r['genuine_acc']):>7} "
              f"{fmt(r['spoof_acc']):>7} {fmt(r['f1']):>6} "
              f"{fmt(r.get('roc_auc')):>6} {fmt(r.get('eer')):>5} "
              f"{fmt(r['fpr']):>5} {fmt(r.get('avg_latency_ms')):>8} "
              f"{fmt(r['size_mb']):>6}")

    print("\nDone. All reports saved to benchmark_results/")
    return all_results, summary_rows

if __name__ == "__main__":
    main()
