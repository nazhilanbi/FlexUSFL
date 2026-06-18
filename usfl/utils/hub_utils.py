import os
from typing import Optional


MODEL_HUB_IDS = {
    "qwen/qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "meta-llama/llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen/qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

DATASET_HUB_IDS = {
    "gsm8k": "openai/gsm8k",
    "dialogsum": "knkarthick/dialogsum",
    "e2e": "GEM/e2e_nlg",
}

DATASET_CONFIGS = {
    "gsm8k": "main",
}


def resolve_model_source(
    model_name: str,
    local_root: str,
    local_path: Optional[str] = None,
) -> str:
    """Prefer a local model directory and otherwise return its Hub ID."""
    if local_path and os.path.isdir(local_path):
        return local_path

    if os.path.isdir(model_name):
        return model_name

    local_candidate = os.path.join(local_root, model_name)
    if os.path.isdir(local_candidate):
        return local_candidate

    return MODEL_HUB_IDS.get(model_name.lower(), model_name)


def resolve_dataset_source(dataset_name: str, local_root: str) -> str:
    """Prefer a local dataset directory and otherwise return its Hub ID."""
    local_candidate = os.path.join(local_root, dataset_name)
    if os.path.isdir(local_candidate):
        return local_candidate

    return DATASET_HUB_IDS.get(dataset_name.lower(), dataset_name)


def get_dataset_config(dataset_name: str) -> Optional[str]:
    """Return the Hugging Face dataset configuration used by the artifact."""
    return DATASET_CONFIGS.get(dataset_name.lower())
