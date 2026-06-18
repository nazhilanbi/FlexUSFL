import torch
import torch.nn as nn
from typing import Dict, Any, List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import transformers
from usfl.llm import (
    load_gpt_server_model,
    load_llama_server,
    load_qwen3_server,
    SplitModelConfig,
)
from usfl.llm import (
    load_gpt_client_models,
    load_llama_client,
    load_qwen3_client,
    SplitModelConfig,
)
from usfl.utils.dataset.base import get_client_dataloaders
from usfl.utils.dataset.exp import get_dataset
from usfl.utils.hub_utils import resolve_model_source
from usfl import env as env_config


def _build_model_load_kwargs(quantization_config=None) -> Dict[str, Any]:
    """Build consistent model-loading arguments for client and server models."""
    load_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "cpu",
        "attn_implementation": "eager",
    }
    if quantization_config is None:
        # Keep the reference artifact in FP32 instead of relying on the
        # Transformers version or the model config's torch_dtype default.
        load_kwargs["torch_dtype"] = torch.float32
    return load_kwargs


def _build_split_lora_config() -> LoraConfig:
    """Create a generic LoRA config for a split model component.

    Split head, server, and tail models are not complete causal language
    models and do not implement generation methods. Leaving ``task_type``
    unset makes PEFT use its generic ``PeftModel`` wrapper.
    """
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )


def get_model_layer_num(model_dir: str) -> int:
    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    else:
        raise ValueError("Cannot find layer number")


def load_client(model_dir: str, client_args: Dict[str, Any], split_point: int = 2):
    model_source = resolve_model_source(
        client_args["model"],
        local_root=env_config.model_root_dir,
        local_path=model_dir,
    )
    if client_args["use_qlora_4bit"] or client_args["use_qlora_8bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=client_args["use_qlora_4bit"],
            load_in_8bit=client_args["use_qlora_8bit"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **_build_model_load_kwargs(quantization_config),
    )

    if client_args["use_qlora_4bit"] or client_args["use_qlora_8bit"]:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.pad_token = tokenizer.eos_token

    split_config = SplitModelConfig(
        head_layer_num=split_point,
        tail_layer_num=split_point,
    )

    if "gpt" in client_args["model"].lower():
        head, tail = load_gpt_client_models(model, split_config)
    elif "llama" in client_args["model"].lower():
        head, tail = load_llama_client(model, split_config)
    elif "qwen" in client_args["model"].lower():
        head, tail = load_qwen3_client(model, split_config)
    else:
        raise ValueError(f"unsupported model card {client_args['model']}")
    if client_args["use_lora"]:
        head = get_peft_model(head, _build_split_lora_config())
        tail = get_peft_model(tail, _build_split_lora_config())
    return head, tail, tokenizer


def load_dataset(
    dataset_name: str = "gsm8k",
    tokenizer: AutoTokenizer = None,
    client_ids: List[int] = [0],
    batch_size: int = 4,
    max_seq_len: int = 256,
    partition_mode: str = "exclusive",
    sample_ratio: float = 0.1,
):
    # usl_dataset = get_dataset(dataset_name=dataset_name, tokenizer=tokenizer, client_ids=client_ids)
    client_dataloaders = get_client_dataloaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        client_ids=client_ids,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        splits=["train", "test"],
        shuffle=False,
        partition_mode=partition_mode,
        sample_ratio=sample_ratio,
    )
    return client_dataloaders


def load_server_model(
    model_dir: str,
    server_args: Dict[str, Any],
    split_point: int = 2,
) -> nn.Module:
    model_source = resolve_model_source(
        server_args["model"],
        local_root=env_config.model_root_dir,
        local_path=model_dir,
    )
    if server_args["use_qlora_4bit"] or server_args["use_qlora_8bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=server_args["use_qlora_4bit"],
            load_in_8bit=server_args["use_qlora_8bit"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **_build_model_load_kwargs(quantization_config),
    )
    model.train()
    if server_args["use_qlora_4bit"] or server_args["use_qlora_8bit"]:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.pad_token = tokenizer.eos_token

    split_config = SplitModelConfig(
        head_layer_num=split_point,
        tail_layer_num=split_point,
    )

    if "gpt" in server_args["model"].lower():
        server = load_gpt_server_model(model, split_config)
    elif "llama" in server_args["model"].lower():
        server = load_llama_server(model, split_config)
    elif "qwen" in server_args["model"].lower():
        server = load_qwen3_server(model, split_config)
    else:
        raise ValueError(f"unsupported model card {server_args['model']}")
    if server_args["use_lora"]:
        server = get_peft_model(server, _build_split_lora_config())
    return server
