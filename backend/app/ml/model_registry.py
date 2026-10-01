from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import HfApi


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    architecture: str
    sample_rate: int = 16000
    requires_mono: bool = True
    description: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ast": ModelSpec(
        key="ast",
        repo_id="MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
        architecture="ASTForAudioClassification",
        sample_rate=16000,
        requires_mono=True,
        description="Baseline AST model used by the existing backend",
    ),
    "w2v2-aasist": ModelSpec(
        key="w2v2-aasist",
        repo_id="W2V2-AASIST",
        architecture="Wav2Vec2-AASIST",
        sample_rate=16000,
        requires_mono=True,
        description="Candidate Wav2Vec2 anti-spoof model",
    ),
    "spectra-aasist3": ModelSpec(
        key="spectra-aasist3",
        repo_id="Spectra-AASIST3",
        architecture="Spectra-AASIST3",
        sample_rate=16000,
        requires_mono=True,
        description="Candidate Spectra-AASIST3 anti-spoof model",
    ),
    "pellav2": ModelSpec(
        key="pellav2",
        repo_id="PellaV2",
        architecture="PellaV2",
        sample_rate=16000,
        requires_mono=True,
        description="Candidate PellaV2 anti-spoof model",
    ),
}


def get_model_spec(model_key: str | None) -> ModelSpec:
    normalized = (model_key or "ast").strip().lower()
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]
    if "/" in normalized:
        for spec in MODEL_REGISTRY.values():
            if spec.repo_id.lower() == normalized.lower():
                return spec
    raise ValueError(f"Unsupported model key: {model_key}")


def list_model_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())


def repo_exists(repo_id: str) -> bool:
    try:
        HfApi().repo_info(repo_id, timeout=20)
        return True
    except Exception:
        return False
