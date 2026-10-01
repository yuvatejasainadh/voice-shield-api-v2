# WavLM Diagnostic Analysis

## TASK 1 & 5 — Verify Inference & Preprocessing
**Model:** `abhishtagatya/wavlm-base-960h-itw-deepfake`
- **Architecture:** `WavLMForSequenceClassification`
- **Expected Sample Rate:** 16,000 Hz
- **Label Mapping:** `{'0': 'bona-fide', '1': 'spoof'}`

Audio was correctly loaded using `librosa` at 16kHz mono and processed through `AutoFeatureExtractor`. The model outputs raw logits which were correctly passed through a softmax function to obtain `Probability 0 (Genuine)` and `Probability 1 (Spoof)`. 

The inference mechanics are mathematically sound. The mapping is correct, and the probabilities exactly match the softmax over the logits.

## TASK 2 — Score Distribution
The model's spoof probabilities are extremely polarized. 
- **Genuine Mean Spoof Score:** ~0.100 (heavily skewed by one false positive at 0.994)
- **Spoof Mean Spoof Score:** ~0.104 (heavily skewed by one true positive at 0.996)
- **Median Genuine Spoof Score:** 0.0014
- **Median Spoof Spoof Score:** 0.0017

The model does not provide a smooth gradient of uncertainty. It acts as a binary step function, outputting extreme confidence (~99.8%) for almost every clip, regardless of whether it is right or wrong.

## TASK 3 — Threshold Sweep
| Threshold | Accuracy | Genuine Acc | Spoof Acc | FPR | FNR | Precision | Recall | F1 |
|-----------|----------|-------------|-----------|-----|-----|-----------|--------|----|
| 0.05 - 0.95 | 0.50 | 0.90 | 0.10 | 0.10 | 0.90 | 0.50 | 0.10 | 0.1667 |

Because the probabilities are polarized to either `< 0.02` or `> 0.99`, shifting the decision threshold anywhere between `0.05` and `0.95` yields exactly the same classification outcomes. The model's poor performance is **not** a threshold or calibration issue.

## TASK 4 — Audio Duration Test
We discovered a critical vulnerability related to audio length. 

The original files average ~6.8 seconds. When evaluated at full length, the model misclassified spoof `0001.wav` as genuine (Spoof Probability: 0.010). However, **when truncated to exactly the first 3 seconds, the model correctly identified it as a spoof (Spoof Probability: 0.997).** Conversely, genuine `0009.wav` (correctly classified at full length) became misclassified as spoof when truncated to 3 seconds.

This indicates the model is highly sensitive to input length. It was likely trained on a dataset of short, uniform utterances (e.g., ASVspoof 2019's ~3-4s clips) and loses the spoofing signal when temporal pooling averages features over longer phonetic sequences.

## TASK 6 — Per-File Error Analysis
- **False Positive:** `0008.wav` (Genuine classified as Spoof). 
  - Duration: 7.7s. 
  - This file showed slightly higher RMS energy (0.056) than the genuine average, but nothing mathematically anomalous.
- **False Negatives:** 9 out of 10 spoof files were misclassified as genuine.
  - The single correctly classified spoof (`0006.wav`) was 5.9 seconds long, but the others ranged from 4.6s to 8.9s. 
  - The model fundamentally defaults to "Genuine" when presented with out-of-distribution audio lengths or environments.

## TASK 7 — Compare with AST
- **AST Baseline:** Accuracy 0.50 (Genuine Acc: 0.00, Spoof Acc: 1.00). 
- **WavLM:** Accuracy 0.50 (Genuine Acc: 0.90, Spoof Acc: 0.10).

Both models failed, but in opposite directions. AST suffered from domain collapse and flagged everything as fake. WavLM suffered from domain collapse and flagged almost everything as real.

## TASK 8 — DO NOT OVERFIT
*Threshold results are exploratory because the evaluation dataset contains only 20 samples.* Tuning a threshold or dynamically truncating audio based on these 20 samples will result in severe overfitting and will not generalize to production traffic.

## TASK 9 — FINAL CONCLUSION
**Conclusion: C (Duration Issue) and D (Dataset/Domain Mismatch).**

The WavLM model exhibits severe domain mismatch exacerbated by an inability to handle variable-length telephony audio. The model acts as a highly polarized step-function rather than outputting a calibrated probability distribution. Because temporal averaging over longer clips washes out the specific artifacts it was trained to detect, it defaults to a "Genuine" classification for 90% of spoofed files.

**Recommendation:**
The WavLM architecture itself (with an EER of 0.20 and fast CPU inference) is highly capable. However, this specific Hugging Face checkpoint (`abhishtagatya/wavlm-base-960h-itw-deepfake`) is genuinely unsuitable for our data out-of-the-box. 

It is worth further investigation **only if** we fine-tune it ourselves using a dataset that matches our actual production audio lengths and acoustic conditions. Do not deploy it currently.
