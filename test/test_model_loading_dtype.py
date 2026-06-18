import unittest

import torch

from usfl.utils.load_utils import _build_model_load_kwargs


class TestModelLoadingDtype(unittest.TestCase):
    def test_non_quantized_models_are_loaded_in_float32(self):
        load_kwargs = _build_model_load_kwargs()

        self.assertEqual(load_kwargs["torch_dtype"], torch.float32)
        self.assertIsNone(load_kwargs["quantization_config"])

    def test_quantized_models_keep_the_quantization_compute_dtype(self):
        quantization_config = object()

        load_kwargs = _build_model_load_kwargs(quantization_config)

        self.assertIs(load_kwargs["quantization_config"], quantization_config)
        self.assertNotIn("torch_dtype", load_kwargs)


if __name__ == "__main__":
    unittest.main()
