# Voice Anti-Spoofing Benchmark

This directory contains the results of an automated local benchmark comparing various Hugging Face voice anti-spoofing and deepfake detection models against an AST baseline. 

## Dataset
- **Total files:** 20 (.wav format)
- **Genuine:** 10 samples
- **Spoof:** 10 samples

## Hardware & Environment
- **Operating System:** Windows
- **Python Version:** 3.12.10
- **PyTorch Version:** 2.4.1+cpu
- **Transformers Version:** 4.44.2
- **Device:** CPU (No CUDA/GPU available)
- **Total System RAM:** 16 GB

## Evaluated Models
1. `SpeechAntiSpoofingBenchmarks/AASIST`
2. `rpeddu/enhanced-rawnet2-antispoofing`
3. `abhishtagatya/wavlm-base-960h-itw-deepfake` (WavLM-based)
4. `garystafford/wav2vec2-deepfake-voice-detector` (Wav2Vec2-based)
5. `WWWxp/wav2vec2_spoof_dection1` (Wav2Vec2-based)
6. `MelodyMachine/Deepfake-audio-detection-V2` (Wav2Vec2-based)

*Note: Models that lack standard Transformers architectures or config.json files (e.g., AASIST, RawNet2 custom checkpoints) were documented as FAILED (Invalid/No Config) in accordance with our requirement to test natively supported models.*

## Methodology
1. **Preprocessing:** Each audio file is loaded at the native sample rate of the model's feature extractor (using `librosa` and `AutoFeatureExtractor`).
2. **Inference:** The model's classification head outputs logits, which are converted to probabilities using softmax. The label with the highest probability is selected as the prediction.
3. **Threshold:** The default decision rule (argmax) is used to calculate accuracy, F1, and confusion matrix metrics.
4. **Metrics:** EER and ROC-AUC are computed based on the spoof class probability.
5. **Latency:** Inference latency is recorded per file, excluding model loading time.

## Reproducibility
To run the benchmark yourself:
1. Ensure `scikit-learn`, `librosa`, and `transformers` are installed in the `backend` environment.
2. Navigate to `backend/`.
3. Run `python run_benchmark.py`.
4. The output will be saved in `backend/benchmark_results/`.

*Note: This benchmark infrastructure does not modify the production detector or change the overall API architecture.*
