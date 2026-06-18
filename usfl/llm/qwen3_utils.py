from typing import Any

import torch


def extract_qwen3_hidden_states(layer_output: Any) -> torch.Tensor:
    """Extract hidden states from Qwen3 decoder outputs across Transformers versions.

    Transformers 4.51 returns a tuple whose first item is the hidden-state
    tensor, while newer versions return the tensor directly.
    """
    if isinstance(layer_output, torch.Tensor):
        return layer_output

    if isinstance(layer_output, (tuple, list)):
        if not layer_output:
            raise ValueError("Qwen3 decoder layer returned an empty sequence")

        hidden_states = layer_output[0]
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states

    raise TypeError(
        "Unsupported Qwen3 decoder layer output type: "
        f"{type(layer_output).__name__}"
    )
