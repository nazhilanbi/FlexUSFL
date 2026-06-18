import unittest

import torch

from usfl.llm.qwen3_utils import extract_qwen3_hidden_states


class TestQwen3DecoderOutputCompatibility(unittest.TestCase):
    def test_accepts_tensor_output_from_new_transformers(self):
        hidden_states = torch.randn(2, 4, 8)

        result = extract_qwen3_hidden_states(hidden_states)

        self.assertIs(result, hidden_states)
        self.assertEqual(result.shape, (2, 4, 8))

    def test_accepts_tuple_output_from_transformers_4_51(self):
        hidden_states = torch.randn(2, 4, 8)

        result = extract_qwen3_hidden_states((hidden_states,))

        self.assertIs(result, hidden_states)

    def test_rejects_empty_or_unknown_outputs(self):
        with self.assertRaises(ValueError):
            extract_qwen3_hidden_states(())

        with self.assertRaises(TypeError):
            extract_qwen3_hidden_states(None)


if __name__ == "__main__":
    unittest.main()
