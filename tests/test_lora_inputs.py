"""Input-contract tests only: stdlib, dummy files, no ML imports or model execution."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "train_sd_lora.py"
SPEC = importlib.util.spec_from_file_location("lora_source_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InputContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        # Existence tests intentionally use non-image contents; no image decoder is invoked.
        (self.dataset / "images" / "one.png").write_bytes(b"dummy file")
        self.metadata = self.dataset / "metadata.jsonl"

    def write_rows(self, rows):
        self.metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8-sig")

    def test_relative_paths_captions_and_repeated_image(self):
        self.write_rows([
            {"file_name": "images/one.png", "text": " first caption "},
            {"file_name": "images/one.png", "text": "second caption"},
        ])
        rows = MODULE.read_manifest(self.metadata)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "first caption")
        self.assertEqual(rows[0]["image"], (self.dataset / "images" / "one.png").resolve())

    def test_bad_caption_fails_before_training(self):
        for caption in ("", "   ", None, 12):
            with self.subTest(caption=caption):
                self.write_rows([{"file_name": "images/one.png", "text": caption}])
                with self.assertRaisesRegex(ValueError, "Line 1.*caption"):
                    MODULE.read_manifest(self.metadata)

    def test_missing_image_and_bad_json_report_line(self):
        self.write_rows([{"file_name": "images/missing.png", "text": "caption"}])
        with self.assertRaisesRegex(ValueError, "Line 1.*not found"):
            MODULE.read_manifest(self.metadata)
        self.metadata.write_text("\n{broken}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Line 2.*invalid JSON"):
            MODULE.read_manifest(self.metadata)

    def test_absolute_path_and_parent_escape_are_rejected(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"dummy file")
        for name in (str(outside.resolve()), "../outside.png"):
            with self.subTest(name=name):
                self.write_rows([{"file_name": name, "text": "caption"}])
                with self.assertRaisesRegex(ValueError, "relative|inside"):
                    MODULE.read_manifest(self.metadata)

    def test_empty_or_non_object_manifest(self):
        for content in (" \n", "[]\n"):
            self.metadata.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.read_manifest(self.metadata)

    def test_prompt_contract(self):
        path = self.root / "prompts.json"
        path.write_text('[" one ", "two"]', encoding="utf-8")
        self.assertEqual(MODULE.read_prompts(path), ["one", "two"])
        for bad in ([], [""], [3], {"prompt": "one"}):
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.read_prompts(path)

    def test_outputs_never_overwrite_existing_run(self):
        output = self.root / "results"
        MODULE.new_output_directory(output)
        marker = output / "run_config.json"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not empty"):
            MODULE.new_output_directory(output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_model_revision_is_pinned_for_later_comparison(self):
        hub = types.ModuleType("huggingface_hub")
        commit = "a" * 40
        hub.model_info = Mock(return_value=types.SimpleNamespace(sha=commit))
        hub.try_to_load_from_cache = Mock(return_value=str(self.root / "snapshots" / commit / "model_index.json"))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(MODULE.resolve_revision("org/base-model", "main", False), commit)
            hub.model_info.assert_called_once_with("org/base-model", revision="main")
            self.assertEqual(MODULE.resolve_revision("org/base-model", None, True), commit)
            self.assertIsNone(MODULE.resolve_revision(str(self.dataset), None, True))
            hub.try_to_load_from_cache.return_value = None
            with self.assertRaisesRegex(ValueError, "cached"):
                MODULE.resolve_revision("org/base-model", None, True)

    def test_cli_rejects_invalid_runs_without_ml_imports(self):
        cases = [
            ["train", "--model", "example", "--metadata", "file", "--output-dir", "out", "--rank", "0"],
            ["train", "--model", "example", "--metadata", "file", "--output-dir", "out", "--resolution", "513"],
            ["compare", "--adapter-dir", "run", "--prompts", "file", "--output-dir", "out", "--seeds", "42", "42"],
            ["compare", "--adapter-dir", "run", "--prompts", "file", "--output-dir", "out", "--lora-scale", "nan"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    MODULE.parse_args(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
