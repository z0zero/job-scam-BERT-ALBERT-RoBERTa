import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.classifier import ScamClassifier


class ScamClassifierTests(unittest.TestCase):
    def test_deployment_disables_streamlit_file_watcher(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / ".streamlit"
            / "config.toml"
        )
        self.assertTrue(config_path.exists(), "Streamlit config is missing")

        with config_path.open("rb") as file:
            config = tomllib.load(file)

        self.assertEqual(config["server"]["fileWatcherType"], "none")

    def test_deployment_uses_tested_cpu_only_model_runtime(self):
        requirements = (
            Path(__file__).resolve().parents[2] / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()

        self.assertFalse(
            any(line.startswith("--extra-index-url") for line in requirements)
        )
        self.assertIn(
            "torch @ https://download-r2.pytorch.org/whl/cpu/"
            "torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
            "#sha256=ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238f"
            "bf5da15a72650b1d ; sys_platform == \"linux\" and "
            "platform_machine == \"x86_64\" and python_version == \"3.12\"",
            requirements,
        )
        self.assertIn(
            "torch @ https://download-r2.pytorch.org/whl/cpu/"
            "torch-2.10.0%2Bcpu-cp312-cp312-win_amd64.whl"
            "#sha256=21cb5436978ef47c823b7a813ff0f8c2892e266cfe0f1d94"
            "4879b5fba81bf4e1 ; sys_platform == \"win32\" and "
            "platform_machine == \"AMD64\" and python_version == \"3.12\"",
            requirements,
        )
        self.assertIn("transformers==5.0.0", requirements)

    def test_hugging_face_network_settings_are_configured_before_import(self):
        env = os.environ.copy()
        env.pop("HF_HUB_DISABLE_XET", None)
        env.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        env.pop("HF_HUB_ETAG_TIMEOUT", None)
        env.pop("HF_HUB_DISABLE_SYMLINKS", None)
        env.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "import src.models.classifier; "
                    "print(os.environ.get('HF_HUB_DISABLE_XET')); "
                    "print(os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')); "
                    "print(os.environ.get('HF_HUB_ETAG_TIMEOUT')); "
                    "print(os.environ.get('HF_HUB_DISABLE_SYMLINKS')); "
                    "print(os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN'))"
                ),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        expected_symlink_setting = "1" if os.name == "nt" else "None"
        self.assertEqual(
            result.stdout.splitlines(),
            ["1", "120", "30", expected_symlink_setting, "1"],
        )

    @patch("src.models.classifier.snapshot_download")
    @patch("src.models.classifier.AutoModelForSequenceClassification")
    @patch("src.models.classifier.AutoTokenizer")
    def test_remote_model_downloads_one_snapshot_then_loads_locally(
        self,
        tokenizer_factory,
        model_factory,
        snapshot_download,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir)
            (snapshot_path / "model_meta.json").write_text(
                '{"model_name": "BERT"}',
                encoding="utf-8",
            )
            snapshot_download.return_value = str(snapshot_path)
            model = Mock()
            model_factory.from_pretrained.return_value = model

            with patch.dict(
                os.environ,
                {
                    "HF_MODEL_ID": "z0zero/job-scam-bert",
                    "HF_TOKEN": "invalid-stale-token",
                },
                clear=False,
            ):
                classifier = ScamClassifier()
                classifier.load_model()

        snapshot_download.assert_called_once_with(
            repo_id="z0zero/job-scam-bert",
            token=False,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "model_meta.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        tokenizer_factory.from_pretrained.assert_called_once_with(
            str(snapshot_path),
            local_files_only=True,
        )
        model_factory.from_pretrained.assert_called_once_with(
            str(snapshot_path),
            local_files_only=True,
        )
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with(classifier.device)
        self.assertEqual(classifier.meta, {"model_name": "BERT"})


if __name__ == "__main__":
    unittest.main()
