import os
import tempfile
import unittest
from unittest.mock import patch

from usfl.utils.dataset import datasets as dataset_module
from usfl.utils.hub_utils import (
    get_dataset_config,
    resolve_dataset_source,
    resolve_model_source,
)


class TestHubFallback(unittest.TestCase):
    def test_models_prefer_local_directories(self):
        with tempfile.TemporaryDirectory() as local_root:
            model_name = "qwen/qwen3-0.6b"
            local_model = os.path.join(local_root, model_name)
            os.makedirs(local_model)

            source = resolve_model_source(model_name, local_root)

            self.assertEqual(source, local_model)

    def test_models_fall_back_to_namespaced_hub_ids(self):
        with tempfile.TemporaryDirectory() as local_root:
            expected_sources = {
                "qwen/qwen3-0.6b": "Qwen/Qwen3-0.6B",
                "meta-llama/llama3.2-1b": "meta-llama/Llama-3.2-1B",
                "qwen/qwen3-1.7b": "Qwen/Qwen3-1.7B",
            }

            for model_name, expected_source in expected_sources.items():
                with self.subTest(model_name=model_name):
                    self.assertEqual(
                        resolve_model_source(model_name, local_root),
                        expected_source,
                    )

    def test_datasets_prefer_local_directories(self):
        with tempfile.TemporaryDirectory() as local_root:
            for dataset_name in ("gsm8k", "dialogsum", "e2e"):
                os.makedirs(os.path.join(local_root, dataset_name))

            for dataset_name in ("gsm8k", "dialogsum", "e2e"):
                with self.subTest(dataset_name=dataset_name):
                    self.assertEqual(
                        resolve_dataset_source(dataset_name, local_root),
                        os.path.join(local_root, dataset_name),
                    )

    def test_datasets_fall_back_to_namespaced_hub_ids(self):
        with tempfile.TemporaryDirectory() as local_root:
            expected_sources = {
                "gsm8k": "openai/gsm8k",
                "dialogsum": "knkarthick/dialogsum",
                "e2e": "GEM/e2e_nlg",
            }

            for dataset_name, expected_source in expected_sources.items():
                with self.subTest(dataset_name=dataset_name):
                    self.assertEqual(
                        resolve_dataset_source(dataset_name, local_root),
                        expected_source,
                    )

        self.assertEqual(get_dataset_config("gsm8k"), "main")
        self.assertIsNone(get_dataset_config("dialogsum"))
        self.assertIsNone(get_dataset_config("e2e"))

    def test_dataset_loader_passes_namespaced_ids_and_config(self):
        with tempfile.TemporaryDirectory() as local_root:
            with (
                patch.object(
                    dataset_module.env_config,
                    "dataset_cache_dir",
                    local_root,
                ),
                patch.object(dataset_module, "load_dataset") as mock_load_dataset,
            ):
                dataset_module._load_artifact_dataset("gsm8k")
                mock_load_dataset.assert_called_once_with(
                    "openai/gsm8k",
                    "main",
                    cache_dir=local_root,
                )

                mock_load_dataset.reset_mock()
                dataset_module._load_artifact_dataset("dialogsum")
                mock_load_dataset.assert_called_once_with(
                    "knkarthick/dialogsum",
                    cache_dir=local_root,
                )

                mock_load_dataset.reset_mock()
                dataset_module._load_artifact_dataset("e2e")
                mock_load_dataset.assert_called_once_with(
                    "GEM/e2e_nlg",
                    cache_dir=local_root,
                )


if __name__ == "__main__":
    unittest.main()
