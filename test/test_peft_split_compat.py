import unittest

import torch
from peft import PeftModel, PeftModelForCausalLM, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from usfl.llm.client.qwen3 import load_qwen3_client
from usfl.llm.server.qwen3 import load_qwen3_server
from usfl.llm.split_config import SplitModelConfig
from usfl.utils.load_utils import _build_split_lora_config


class TestPeftSplitModelCompatibility(unittest.TestCase):
    def setUp(self):
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            _attn_implementation="eager",
        )
        model = Qwen3ForCausalLM(config)
        split_config = SplitModelConfig(head_layer_num=1, tail_layer_num=1)
        self.head, self.tail = load_qwen3_client(model, split_config)
        self.server = load_qwen3_server(model, split_config)

    def test_split_lora_uses_generic_peft_wrapper(self):
        head = get_peft_model(self.head, _build_split_lora_config())
        tail = get_peft_model(self.tail, _build_split_lora_config())
        server = get_peft_model(self.server, _build_split_lora_config())

        for component in (head, tail, server):
            self.assertIsInstance(component, PeftModel)
            self.assertNotIsInstance(component, PeftModelForCausalLM)
            self.assertIsNone(component.peft_config["default"].task_type)

    def test_split_lora_forward_and_backward(self):
        head = get_peft_model(self.head, _build_split_lora_config())
        tail = get_peft_model(self.tail, _build_split_lora_config())
        server = get_peft_model(self.server, _build_split_lora_config())
        input_ids = torch.randint(1, 128, (2, 5))
        attention_mask = torch.ones_like(input_ids)

        hidden_states, causal_mask, position_embeddings = head(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = server(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
        )
        output = tail(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            labels=input_ids,
        )
        output.loss.backward()

        trainable_parameters = [
            parameter
            for component in (head, server, tail)
            for name, parameter in component.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        self.assertTrue(trainable_parameters)
        self.assertTrue(any(parameter.grad is not None for parameter in trainable_parameters))


if __name__ == "__main__":
    unittest.main()
