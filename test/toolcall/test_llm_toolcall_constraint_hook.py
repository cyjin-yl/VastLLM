import json
import sys
import unittest
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_FASTLLM_PYTOOLS = REPO_ROOT / "tools" / "fastllm_pytools"
_FASTLLM_BUILD_PACKAGE = REPO_ROOT / "build" / "tools" / "ftllm"
_NATIVE_LIBRARY_NAMES = (
    "libfastllm_tools.so",
    "libfastllm_tools-cu11.so",
    "libfastllm_tools-cpu.so",
    "fastllm_tools.dll",
    "libfastllm_tools.dylib",
)


def _import_llm_or_skip():
    source_native = any((_FASTLLM_PYTOOLS / name).exists()
                        for name in _NATIVE_LIBRARY_NAMES)
    build_native = any((_FASTLLM_BUILD_PACKAGE / name).exists()
                       for name in _NATIVE_LIBRARY_NAMES)
    if not source_native and not build_native:
        raise unittest.SkipTest(
            "fastllm native library is not available for llm.py hook tests")
    try:
        if source_native:
            from tools.fastllm_pytools import llm as llm_module
        else:
            build_tools = str(_FASTLLM_BUILD_PACKAGE.parent)
            if build_tools not in sys.path:
                sys.path.insert(0, build_tools)
            from ftllm import llm as llm_module
    except SystemExit as exc:
        raise unittest.SkipTest(
            "fastllm native library could not be loaded for llm.py hook tests"
        ) from exc
    return llm_module


class _NativeSetter:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def __call__(self, model_id, payload):
        self.calls.append((model_id, payload))
        return self.result


class _FakeNativeLib:
    def __init__(self, setter=None):
        if setter is not None:
            self.set_tool_call_constraint_llm_model = setter


class LlmToolCallConstraintHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_module = _import_llm_or_skip()

    def _model(self):
        model = self.llm_module.model.__new__(self.llm_module.model)
        model.model = 123
        return model

    def test_native_tool_call_constraint_setter_receives_json(self):
        setter = _NativeSetter(result=True)
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib(setter)
        constraint = {
            "name_constraint": {
                "type": "tool_name_enum",
                "allowed_names": ["get_weather"],
            }
        }
        try:
            applied = self._model()._apply_tool_call_constraint_to_native(
                constraint)
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertTrue(applied)
        self.assertEqual(len(setter.calls), 1)
        model_id, payload = setter.calls[0]
        self.assertEqual(model_id, 123)
        self.assertEqual(json.loads(payload.decode()), constraint)

    def test_native_tool_call_constraint_setter_clears_on_none(self):
        setter = _NativeSetter(result=True)
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib(setter)
        try:
            applied = self._model()._apply_tool_call_constraint_to_native(None)
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertTrue(applied)
        self.assertEqual(len(setter.calls), 1)
        model_id, payload = setter.calls[0]
        self.assertEqual(model_id, 123)
        self.assertIsNone(json.loads(payload.decode()))

    def test_gguf_ori_config_applies_qwen35_float16_atype(self):
        class FakeNativeLib:
            def __init__(self):
                self.atype_calls = []
                self.kv_dtype_calls = []

            def create_llm_model_from_gguf(self, path, ori_path):
                return 123

            def set_model_atype(self, model_id, atype):
                self.atype_calls.append((model_id, atype))

            def set_model_kv_cache_dtype(self, model_id, dtype):
                self.kv_dtype_calls.append((model_id, dtype))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gguf_path = root / "model.gguf"
            gguf_path.touch()
            ori_path = root / "ori"
            ori_path.mkdir()
            (ori_path / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }), encoding="utf-8")

            native = FakeNativeLib()
            original_lib = self.llm_module.fastllm_lib
            original_tokenizer_loader = self.llm_module.try_load_hf_tokenizer
            self.llm_module.fastllm_lib = native
            self.llm_module.try_load_hf_tokenizer = lambda path: None
            try:
                model = self.llm_module.model(
                    str(gguf_path), dtype="float16",
                    kv_cache_dtype="float16",
                    ori_model_path=str(ori_path))
            finally:
                self.llm_module.fastllm_lib = original_lib
                self.llm_module.try_load_hf_tokenizer = original_tokenizer_loader

        self.assertEqual(model.config["model_type"], "qwen3_5")
        self.assertEqual(native.atype_calls, [(123, b"float16")])
        self.assertEqual(native.kv_dtype_calls, [(123, b"float16")])

    def test_missing_native_setter_reports_not_applied(self):
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib()
        try:
            with self.assertLogs(level="DEBUG") as logs:
                applied = self._model()._apply_tool_call_constraint_to_native(
                    {"name_constraint": {"allowed_names": ["get_weather"]}})
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertFalse(applied)
        self.assertIn("does not expose set_tool_call_constraint_llm_model",
                      "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
