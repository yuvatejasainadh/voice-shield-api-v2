# Voice Anti-Spoofing Benchmark Report

## Executive Summary

**Dataset:**
- 20 files
- 10 genuine
- 10 spoof

**AST baseline:**
- 50% accuracy, previously tested (predicted all samples as spoof)

**Conclusion:** 
No model is currently suitable for integration.

While multiple models were successfully benchmarked, their out-of-the-box accuracy is extremely poor on our dataset. Most models collapsed to predicting a single class (either 100% Genuine or 100% Spoof). Although threshold analysis indicates that some models (e.g., `abhishtagatya/wavlm-base-960h-itw-deepfake` with an EER of 0.2) carry a strong underlying signal, their default thresholds are severely miscalibrated for our data.

## Model Comparison Table

| Model | Status | Accuracy | Genuine Acc | Spoof Acc | F1 | ROC-AUC | EER | Avg Latency | P95 Latency | Model Size |
|------|--------|----------|-------------|-----------|----|---------|-----|-------------|-------------|------------|
| AST (Baseline) | PREVIOUSLY TESTED - NOT RERUN | 0.50 | 0.00 | 1.00 | 0.6667 | N/A | N/A | N/A | N/A | N/A |
| SpeechAntiSpoofingBenchmarks/AASIST | FAILED (Invalid/No Config) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| rpeddu/enhanced-rawnet2-antispoofing | FAILED (Invalid/No Config) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| abhishtagatya/wavlm-base-960h-itw-deepfake | PASS | 0.50 | 0.90 | 0.10 | 0.1667 | 0.64 | 0.20 | 425 ms | 573 ms | 360.8 MB |
| garystafford/wav2vec2-deepfake-voice-detector | PASS | 0.50 | 1.00 | 0.00 | 0.0000 | 0.18 | 0.90 | 1824 ms | 3059 ms | 1204.3 MB |
| WWWxp/wav2vec2_spoof_dection1 | PASS | 0.50 | 1.00 | 0.00 | 0.0000 | 0.47 | 0.40 | 728 ms | 1052 ms | 360.8 MB |
| MelodyMachine/Deepfake-audio-detection-V2 | PASS | 0.50 | 1.00 | 0.00 | 0.0000 | 0.32 | 0.60 | 748 ms | 1012 ms | 360.8 MB |

## Detailed Analysis

### Out-of-the-box Performance
- **AST (Baseline):** Erred on the side of extreme caution by predicting everything as spoof. This leads to a 100% false-positive rate (genuine calls blocked).
- **garystafford, WWWxp, MelodyMachine:** These Wav2Vec2 variants erred completely in the opposite direction. They predicted every single audio clip as genuine, leading to a 100% false-negative rate (0 spoof recall).
- **abhishtagatya/wavlm-base-960h-itw-deepfake:** Slightly better default calibration, getting 90% of genuine calls right, but missing 90% of spoofed calls.

### Threshold Analysis
The `abhishtagatya/wavlm-base-960h-itw-deepfake` model showed a ROC-AUC of 0.64 and an EER of 0.20. 
An EER (Equal Error Rate) of 0.20 implies that if we tuned the decision threshold specifically for our dataset, we could achieve approximately 80% accuracy (with a 20% False Positive Rate and 20% False Negative Rate). However, our primary security concern is detecting spoofed/AI-generated voices while avoiding *excessive* false positives on genuine calls. A 20% FPR is still quite high for a production communications system, and tuning on only 20 samples is highly susceptible to overfitting.

### Latency and Hardware
- The benchmark was executed locally on CPU (`Device: cpu`).
- The `abhishtagatya/wavlm-base-960h-itw-deepfake` model was the fastest and lightest, running in roughly **425 ms** per clip (clips averaged ~6.84s in length) with a size of **360 MB**. This is very fast and would suit a local prototype.
- The `garystafford` model is over 1.2 GB and took nearly 2 seconds per inference on CPU.

## Recommendation

**No model is currently suitable for integration.**

Although `abhishtagatya/wavlm-base-960h-itw-deepfake` provides the strongest underlying spoof detection signal (EER=0.20) and excellent CPU inference speed, its default threshold is entirely uncalibrated for our use case, misclassifying 90% of spoofs as genuine. Furthermore, calibrating a threshold based on a 20-file prototype dataset is strongly discouraged for a production-ready anti-spoofing system.

Before replacing the detector, we should either:
1. Gather a much larger, representative dataset to securely tune the decision threshold of the WavLM model.
2. Continue researching models trained on closer acoustic conditions to our telephony/prototype audio. 

As instructed, I have **not** modified the production configuration (`DETECTOR_MODEL_ID` or risk thresholds). All tests were kept strictly isolated to the benchmark environment.