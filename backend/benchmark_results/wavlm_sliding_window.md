# WavLM Sliding-Window Experiment Report

> **EXPLORATORY ANALYSIS — NOT FOR PRODUCTION**
> All threshold and aggregation results below are derived from only 20 audio samples.
> Do NOT select a production threshold or integrate any model based on these results alone.

---

## Configuration

- **Model:** `abhishtagatya/wavlm-base-960h-itw-deepfake`
- **Architecture:** `WavLMForSequenceClassification`
- **Preprocessing:** mono, 16kHz, `AutoFeatureExtractor`
- **Label mapping:** `0 = bona-fide`, `1 = spoof`
- **Window size:** 3.0 seconds
- **Hop size:** 0.5 seconds
- **Short-file handling:** Files shorter than 3s zero-padded to exactly 3s

---

## 1. Full-Length vs Sliding-Window Results Summary

| Method | Accuracy | Genuine Acc | Spoof Acc | FPR | FNR | F1 | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|---|
| **FULL_LENGTH** | 0.50 | 0.90 | 0.10 | 0.10 | 0.90 | 0.1667 | 0.64 | 0.20 |
| **MAX** | 0.60 | 0.50 | 0.70 | 0.50 | 0.30 | 0.6364 | 0.73 | 0.20 |
| **MEAN** | 0.60 | 1.00 | 0.20 | 0.00 | 0.80 | 0.3333 | 0.69 | 0.40 |
| **TOP2_MEAN** | 0.60 | 0.50 | 0.70 | 0.50 | 0.30 | 0.6364 | 0.70 | 0.40 |
| **TOP25_MEAN** | **0.65** | 0.60 | **0.70** | 0.40 | 0.30 | **0.6667** | 0.65 | 0.40 |
| **MAJORITY_VOTE** | 0.60 | 1.00 | 0.20 | 0.00 | 0.80 | 0.3333 | N/A | N/A |

**Best aggregation method:** TOP25_MEAN — achieves 65% accuracy and 70% spoof recall at 0.4 FPR.
**Best spoof recall:** MAX and TOP2_MEAN (0.70 spoof accuracy).
**Best ROC-AUC:** MAX at 0.73 (vs 0.64 full-length — a clear improvement).
**Best F1:** TOP25_MEAN at 0.6667.

### Key improvement over full-length
The sliding-window approach dramatically improves spoof recall from **10% → 70%** by allowing the model to find the specific 3-second window within a longer recording that contains detectable spoofing artifacts.

---

## 2. Per-File Comparison Table (Section 6)

| Filename | GT | Full Score | Max Win | Mean Win | Top2 Mean | Top25 Mean | Majority | Full Pred | Max Pred |
|---|---|---|---|---|---|---|---|---|---|
| 0001.wav | genuine | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0 | genuine | genuine |
| 0002.wav | genuine | 0.0015 | 0.0018 | 0.0015 | 0.0017 | 0.0018 | 0 | genuine | genuine |
| 0003.wav | genuine | 0.0015 | 0.0044 | 0.0019 | 0.0036 | 0.0032 | 0 | genuine | genuine |
| 0004.wav | genuine | 0.0014 | 0.0021 | 0.0017 | 0.0020 | 0.0021 | 0 | genuine | genuine |
| **0005.wav** | genuine | 0.0014 | **0.9961** | 0.0739 | 0.5056 | 0.3390 | 0 | genuine | **spoof ✗** |
| 0006.wav | genuine | 0.0014 | 0.0042 | 0.0019 | 0.0030 | 0.0042 | 0 | genuine | genuine |
| **0007.wav** | genuine | 0.0017 | **0.9981** | 0.3298 | 0.9978 | 0.9978 | 0 | genuine | **spoof ✗** |
| **0008.wav** | genuine | 0.9941 | **0.9982** | 0.2240 | 0.9981 | 0.9981 | 0 | spoof ✗ | **spoof ✗** |
| **0009.wav** | genuine | 0.0019 | **0.9013** | 0.1466 | 0.7220 | 0.7220 | 0 | genuine | **spoof ✗** |
| **0010.wav** | genuine | 0.0014 | **0.9964** | 0.2120 | 0.8479 | 0.8479 | 0 | genuine | **spoof ✗** |
| **0001.wav** | spoof | 0.0104 | **0.9974** | 0.4977 | 0.9960 | 0.9974 | 0 | genuine | **spoof ✓** |
| 0002.wav | spoof | 0.0017 | 0.0039 | 0.0019 | 0.0030 | 0.0030 | 0 | genuine | genuine ✗ |
| 0003.wav | spoof | 0.0014 | 0.0028 | 0.0017 | 0.0023 | 0.0023 | 0 | genuine | genuine ✗ |
| **0004.wav** | spoof | 0.0233 | **0.9984** | 0.5547 | 0.9982 | 0.9982 | 1 | genuine | **spoof ✓** |
| **0005.wav** | spoof | 0.0014 | **0.9970** | 0.1142 | 0.6174 | 0.6174 | 0 | genuine | **spoof ✓** |
| 0006.wav | spoof | 0.9964 | 0.9986 | 0.6008 | 0.9985 | 0.9986 | 1 | spoof ✓ | spoof ✓ |
| **0007.wav** | spoof | 0.0019 | **0.9972** | 0.1692 | 0.9963 | 0.6699 | 0 | genuine | **spoof ✓** |
| 0008.wav | spoof | 0.0014 | 0.0037 | 0.0019 | 0.0032 | 0.0032 | 0 | genuine | genuine ✗ |
| **0009.wav** | spoof | 0.0021 | **0.9987** | 0.4286 | 0.9987 | 0.9987 | 0 | genuine | **spoof ✓** |
| **0010.wav** | spoof | 0.0014 | **0.9986** | 0.1310 | 0.8432 | 0.5629 | 0 | genuine | **spoof ✓** |

### Spoof files MISSED by full-length but DETECTED by at least one 3-second window (MAX):
- `0001.wav` (spoof): full=0.0104 → max_win=**0.9974** ✓
- `0004.wav` (spoof): full=0.0233 → max_win=**0.9984** ✓
- `0005.wav` (spoof): full=0.0014 → max_win=**0.9970** ✓
- `0007.wav` (spoof): full=0.0019 → max_win=**0.9972** ✓
- `0009.wav` (spoof): full=0.0021 → max_win=**0.9987** ✓
- `0010.wav` (spoof): full=0.0014 → max_win=**0.9986** ✓

### Persistent False Negatives (spoof NOT detected even by MAX window):
- `0002.wav` (spoof): full=0.0017, max_win=0.0039 ✗
- `0003.wav` (spoof): full=0.0014, max_win=0.0028 ✗
- `0008.wav` (spoof): full=0.0014, max_win=0.0037 ✗

### Genuine files that became FALSE POSITIVES under MAX window:
- `0005.wav` (genuine): full=0.0014 → max_win=**0.9961** ✗
- `0007.wav` (genuine): full=0.0017 → max_win=**0.9981** ✗
- `0008.wav` (genuine): full=0.9941 → max_win=**0.9982** ✗ (was already FP at full length)
- `0009.wav` (genuine): full=0.0019 → max_win=**0.9013** ✗
- `0010.wav` (genuine): full=0.0014 → max_win=**0.9964** ✗

---

## 3. Threshold Analysis for MAX Aggregation (EXPLORATORY)

> **WARNING — Threshold results are exploratory because the evaluation dataset contains only 20 samples.**
> Do NOT select a production threshold from these results.

All thresholds 0.05–0.90 yield identical results (same MAX polarization):

| Threshold | Accuracy | Genuine Acc | Spoof Acc | FPR | FNR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.05–0.90 | 0.60 | 0.50 | 0.70 | 0.50 | 0.30 | 0.5833 | 0.70 | 0.6364 |
| **0.95** | **0.65** | **0.60** | **0.70** | **0.40** | **0.30** | **0.6364** | **0.70** | **0.6667** |

At threshold ≥0.95, the FPR drops from 0.50 to 0.40 with no change in spoof recall (0.70). Even so, this is still a high FPR for production.

---

## 4. Latency Analysis

| Metric | Value |
|---|---|
| Average window inference | 250.9 ms |
| Median window inference | 252.4 ms |
| P95 window inference | 283.1 ms |
| Average windows per file | 9.1 windows |
| Average total latency per file | 2,813 ms |
| **Estimated latency for 10-second call** | **~3,764 ms** |

A 10-second audio clip generates ~15 windows (= (10–3)/0.5 + 1). At 250ms per window, that is approximately **3.76 seconds of CPU compute** per 10-second clip. This is notably slower than real-time (3.76s CPU time for 10s audio ≈ 0.37x real-time ratio).

---

## 5. Prototype Architecture Analysis

```
Call audio (telephony)
       ↓
  3-second rolling window (hop=0.5s)
       ↓
    WavLM (CPU)
       ↓
  spoof score per window
       ↓
  TOP25_MEAN or MAX aggregation
       ↓
  suspicious / not suspicious
```

**Detection quality:** Improved from 10% to 70% spoof recall using MAX aggregation. Three spoof files remain completely undetected.
**False positives:** 5 out of 10 genuine files are now flagged as spoof under MAX. This is a 50% FPR — unacceptably high for a security product that should not block legitimate callers.
**False negatives:** 3 out of 10 spoof files are not detected even with windowing (`0002.wav`, `0003.wav`, `0008.wav`). These appear to be either: (a) high-quality spoofs that do not generate localized artifacts in any 3s window, or (b) out-of-distribution audio the model was never trained to detect.
**Latency:** ~3.76 seconds to process a 10-second call on CPU. This is workable for an asynchronous post-call analysis pipeline, but too slow for real-time interception.
**CPU requirements:** ~250ms per 3s window on a modern CPU. Roughly 3–4 GB RAM. An 8-core CPU could theoretically parallelize windows, reducing wall-clock time significantly.
**Scalability:** Not scalable at >50 concurrent calls on a single CPU machine. Would require batching or GPU.

---

## 6. Final Decision

**C. More data is required before deciding.**

The sliding-window approach shows real and measurable improvement: spoof recall improved from 10% to 70%, and ROC-AUC improved from 0.64 to 0.73. Six out of nine previously-missed spoof files were correctly recovered by finding the most suspicious 3-second window.

However, we cannot confidently recommend it because:

1. **50% FPR is unacceptably high.** Half of genuine callers would receive a false alarm. This is a critical failure for a voice security product.
2. **3 spoof files remain undetected at all.** These appear to be qualitatively different from the detected spoofs. Without knowing the nature of the spoof content, we cannot attribute this to domain mismatch vs. model limitation.
3. **20 samples is statistically insufficient.** The FPR difference of 10 percentage points at the 0.95 threshold (0.50→0.40) represents just 1 genuine file. With only 10 genuine samples, all FPR numbers have ±10% granularity at minimum.

The model is architecturally capable and shows strong localized spoof detection in targeted windows. We recommend: gather at minimum 100–200 labeled samples before reaching a deployment decision, and evaluate whether the 3 persistent false-negative spoofs represent a systematic failure mode.

---

## Report Files

| File | Description |
|---|---|
| `wavlm_sliding_window_scores.csv` | Per-window inference results |
| `wavlm_sliding_window_results.csv` | Per-aggregation metrics |
| `wavlm_sliding_window_thresholds.csv` | MAX threshold sweep |
| `wavlm_sliding_window.json` | Full structured JSON output |
| `wavlm_sliding_window_distribution.png` | Score distributions |
