from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_core():
    module_name = "remap_lora_28_to_40_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "remap_lora_28_to_40.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_comfy_stubs(lora_root: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda category: ["a.safetensors", "b.safetensors"]

    def get_full_path_or_raise(category, name):
        path = lora_root / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return str(path)

    folder_paths.get_full_path_or_raise = get_full_path_or_raise

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy.sd = comfy_sd
    comfy.utils = comfy_utils

    return {
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.sd": comfy_sd,
        "comfy.utils": comfy_utils,
    }, comfy_sd, comfy_utils


def load_backend(lora_root: Path):
    stubs, comfy_sd, comfy_utils = install_comfy_stubs(lora_root)
    package_name = "anima_stack_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    modules = {**stubs, package_name: package}

    with patch.dict(sys.modules, modules):
        core_name = f"{package_name}.remap_lora_28_to_40"
        core_spec = importlib.util.spec_from_file_location(
            core_name, ROOT / "remap_lora_28_to_40.py"
        )
        core_module = importlib.util.module_from_spec(core_spec)
        sys.modules[core_name] = core_module
        assert core_spec.loader is not None
        core_spec.loader.exec_module(core_module)

        backend_name = f"{package_name}.anima_lora_stack"
        backend_spec = importlib.util.spec_from_file_location(
            backend_name, ROOT / "anima_lora_stack.py"
        )
        backend_module = importlib.util.module_from_spec(backend_spec)
        sys.modules[backend_name] = backend_module
        assert backend_spec.loader is not None
        backend_spec.loader.exec_module(backend_module)

    return backend_module, comfy_sd, comfy_utils


class RemapCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_complete_old_to_new_mapping(self):
        self.assertEqual(len(self.core.OLD_TO_NEW), 28)
        self.assertEqual(self.core.OLD_TO_NEW[0], 0)
        self.assertEqual(self.core.OLD_TO_NEW[2], 3)
        self.assertEqual(self.core.OLD_TO_NEW[14], 20)
        self.assertEqual(self.core.OLD_TO_NEW[27], 39)
        self.assertTrue(
            set(self.core.OLD_TO_NEW.values()).isdisjoint(self.core.INSERTION_POSITIONS)
        )

    def test_remap_does_not_create_inserted_layer_keys(self):
        state = {
            f"lora_unet_blocks_{index}_self_attn_q_proj.lora_down.weight": object()
            for index in range(28)
        }
        remapped = self.core.remap_lora_state_dict(state, source_name="test")
        mapped_indices = {
            self.core.find_main_block(key)[1]
            for key in remapped
            if self.core.find_main_block(key)[1] is not None
        }
        self.assertEqual(len(remapped), len(state))
        self.assertTrue(mapped_indices.isdisjoint(self.core.INSERTION_POSITIONS))

    def test_passthrough_key_is_preserved(self):
        passthrough = object()
        state = {
            "lora_unet_blocks_0_self_attn_q_proj.alpha": object(),
            "ss.some_global_value": passthrough,
        }
        remapped = self.core.remap_lora_state_dict(state)
        self.assertIs(remapped["ss.some_global_value"], passthrough)

    def test_invalid_high_layer_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "不支持的主干层 28"):
            self.core.remap_lora_state_dict(
                {"lora_unet_blocks_28_self_attn_q_proj.alpha": object()},
                source_name="invalid.safetensors",
            )

    def test_missing_main_layer_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "未包含可识别"):
            self.core.remap_lora_state_dict({"metadata.only": object()})

    def test_collision_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "键名冲突"):
            self.core.remap_lora_state_dict(
                {
                    "lora_unet_blocks_0_x": object(),
                    "lora_unet_blocks_00_x": object(),
                }
            )


class BackendTests(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend, self.comfy_sd, self.comfy_utils = load_backend(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_skips_disabled_empty_and_zero(self):
        parsed = self.backend._parse_lora_inputs(
            {
                "lora_4": {"on": True, "lora": "b.safetensors", "strength": 0.5},
                "lora_1": {"on": False, "lora": "off.safetensors", "strength": 1},
                "lora_2": {"on": True, "lora": "", "strength": 1},
                "lora_3": {"on": True, "lora": "zero.safetensors", "strength": 0},
            }
        )
        self.assertEqual(parsed, [(4, "b.safetensors", 0.5)])

    def test_multiple_loras_apply_in_numeric_order_and_clip_passthrough(self):
        for name in ("a.safetensors", "b.safetensors"):
            (self.root / name).write_bytes(b"stub")

        states = {
            str(self.root / "a.safetensors"): {
                "lora_unet_blocks_0_self_attn_q_proj.alpha": "a"
            },
            str(self.root / "b.safetensors"): {
                "lora_unet_blocks_1_self_attn_q_proj.alpha": "b"
            },
        }
        self.comfy_utils.load_torch_file = lambda path, **kwargs: (
            states[path],
            {"path": path},
        )
        calls = []

        def apply(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
            value = next(iter(lora.values()))
            calls.append((value, strength_model, strength_clip, clip))
            return [*model, value], clip

        self.comfy_sd.load_lora_for_models = apply
        node = self.backend.Anima28To40PowerLoraStack()
        clip = object()
        model, returned_clip = node.load_loras(
            [],
            clip,
            lora_10={"on": True, "lora": "b.safetensors", "strength": 0.25},
            lora_2={"on": True, "lora": "a.safetensors", "strength": 0.75},
        )

        self.assertEqual(model, ["a", "b"])
        self.assertEqual([call[0] for call in calls], ["a", "b"])
        self.assertTrue(all(call[2] == 0 for call in calls))
        self.assertTrue(all(call[3] is None for call in calls))
        self.assertIs(returned_clip, clip)

    def test_cache_invalidates_when_file_changes(self):
        path = self.root / "a.safetensors"
        path.write_bytes(b"first")
        load_count = 0

        def load_file(file_path, **kwargs):
            nonlocal load_count
            load_count += 1
            return ({"lora_unet_blocks_0_x.alpha": load_count}, None)

        self.comfy_utils.load_torch_file = load_file
        self.comfy_sd.load_lora_for_models = (
            lambda model, clip, lora, sm, sc, **kwargs: (model, clip)
        )
        node = self.backend.Anima28To40PowerLoraStack()
        kwargs = {
            "lora_1": {"on": True, "lora": "a.safetensors", "strength": 1.0}
        }

        node.load_loras(object(), None, **kwargs)
        node.load_loras(object(), None, **kwargs)
        self.assertEqual(load_count, 1)

        path.write_bytes(b"second-version")
        future = time.time_ns() + 10_000_000
        os.utime(path, ns=(future, future))
        node.load_loras(object(), None, **kwargs)
        self.assertEqual(load_count, 2)


if __name__ == "__main__":
    unittest.main()
