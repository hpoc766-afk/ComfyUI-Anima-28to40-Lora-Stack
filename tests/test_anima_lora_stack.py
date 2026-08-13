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

    def load_lora_for_models(
        model, clip, lora, strength_model, strength_clip, lora_metadata=None
    ):
        return model, clip

    def load_bypass_lora_for_models(model, clip, lora, strength_model, strength_clip):
        return model, clip

    comfy_sd.load_lora_for_models = load_lora_for_models
    comfy_sd.load_bypass_lora_for_models = load_bypass_lora_for_models
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

        hook_name = f"{package_name}.auto_remap_hook"
        hook_spec = importlib.util.spec_from_file_location(
            hook_name, ROOT / "auto_remap_hook.py"
        )
        hook_module = importlib.util.module_from_spec(hook_spec)
        sys.modules[hook_name] = hook_module
        assert hook_spec.loader is not None
        hook_spec.loader.exec_module(hook_module)

        backend_name = f"{package_name}.anima_lora_stack"
        backend_spec = importlib.util.spec_from_file_location(
            backend_name, ROOT / "anima_lora_stack.py"
        )
        backend_module = importlib.util.module_from_spec(backend_spec)
        sys.modules[backend_name] = backend_module
        assert backend_spec.loader is not None
        backend_spec.loader.exec_module(backend_module)

    return backend_module, hook_module, comfy_sd, comfy_utils


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

    def test_diffusion_model_dot_format_is_remapped(self):
        value = object()
        state = {
            "diffusion_model.blocks.14.cross_attn.q_proj.lora_A.weight": value,
        }
        remapped = self.core.remap_lora_state_dict(state, source_name="dot-format")
        self.assertNotIn(
            "diffusion_model.blocks.14.cross_attn.q_proj.lora_A.weight",
            remapped,
        )
        self.assertIs(
            remapped["diffusion_model.blocks.20.cross_attn.q_proj.lora_A.weight"],
            value,
        )

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


class AutoRemapHookTests(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend, self.hook, self.comfy_sd, self.comfy_utils = load_backend(
            self.root
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def make_model(*, image_model="anima", block_count=40, config_count=40):
        config = types.SimpleNamespace(
            unet_config={"image_model": image_model, "num_blocks": config_count}
        )
        diffusion_model = types.SimpleNamespace(blocks=[object()] * block_count)
        base_model = types.SimpleNamespace(
            model_config=config, diffusion_model=diffusion_model
        )
        return types.SimpleNamespace(model=base_model)

    def test_model_detection_requires_anima_and_40_blocks(self):
        self.assertTrue(self.hook.is_anima_40_model(self.make_model()))
        self.assertFalse(
            self.hook.is_anima_40_model(self.make_model(image_model="flux"))
        )
        self.assertFalse(
            self.hook.is_anima_40_model(self.make_model(block_count=28))
        )
        self.assertFalse(self.hook.is_anima_40_model(None))
        self.assertFalse(self.hook.is_anima_40_model(object()))

    def test_model_detection_falls_back_to_config_for_wrapped_blocks(self):
        model = self.make_model()
        model.model.diffusion_model = types.SimpleNamespace()
        self.assertTrue(self.hook.is_anima_40_model(model))

    def test_model_detection_handles_hostile_wrappers_conservatively(self):
        class HostileWrapper:
            def __getattr__(self, name):
                raise RuntimeError(f"blocked attribute: {name}")

        self.assertFalse(self.hook.is_anima_40_model(HostileWrapper()))

        model = self.make_model()
        model.model.diffusion_model = HostileWrapper()
        self.assertTrue(self.hook.is_anima_40_model(model))

    def test_28_layer_lora_is_remapped_without_copying_values(self):
        value = object()
        passthrough = object()
        state = {
            "lora_unet_blocks_27_x.lora_down.weight": value,
            "ss.global": passthrough,
        }
        prepared = self.hook.prepare_lora_for_anima_40(
            state, metadata={"filename": "source.safetensors"}
        )
        self.assertIsNot(prepared, state)
        self.assertIs(prepared["lora_unet_blocks_39_x.lora_down.weight"], value)
        self.assertIs(prepared["ss.global"], passthrough)

    def test_lazy_view_reuses_source_mapping_and_exposes_remapped_keys(self):
        value = object()
        state = {
            "lora_unet_blocks_2_x.lora_down.weight": value,
            "ss.global": object(),
        }
        prepared = self.hook.prepare_lora_for_anima_40(state)

        self.assertIsInstance(prepared, self.hook.RemappedLoraView)
        self.assertIs(prepared._source, state)
        self.assertEqual(len(prepared), len(state))
        self.assertEqual(
            set(prepared),
            {"lora_unet_blocks_3_x.lora_down.weight", "ss.global"},
        )
        self.assertIs(
            prepared.get("lora_unet_blocks_3_x.lora_down.weight"),
            value,
        )
        self.assertNotIn("lora_unet_blocks_2_x.lora_down.weight", prepared)

    def test_lazy_view_detects_leading_zero_collision(self):
        with self.assertRaisesRegex(self.hook.LoraRemapError, "collision.safetensors"):
            self.hook.prepare_lora_for_anima_40(
                {
                    "lora_unet_blocks_0_x.alpha": object(),
                    "lora_unet_blocks_00_x.alpha": object(),
                },
                metadata={"filename": "collision.safetensors"},
            )

    def test_key_cache_is_bounded_and_keeps_no_state_dict_reference(self):
        self.hook._analyze_lora_key.cache_clear()
        for index in range(self.hook._KEY_CACHE_SIZE + 64):
            self.hook._analyze_lora_key(
                f"lora_unet_blocks_0_unique_{index}.lora_down.weight"
            )
        info = self.hook._analyze_lora_key.cache_info()
        self.assertEqual(info.maxsize, self.hook._KEY_CACHE_SIZE)
        self.assertEqual(info.currsize, self.hook._KEY_CACHE_SIZE)

    def test_diffusion_model_dot_format_is_classified_and_remapped(self):
        value = object()
        state = {
            "diffusion_model.blocks.27.mlp.layer1.lora_B.weight": value,
        }
        prepared = self.hook.prepare_lora_for_anima_40(state)
        self.assertIs(
            prepared["diffusion_model.blocks.39.mlp.layer1.lora_B.weight"],
            value,
        )

    def test_native_40_layer_lora_is_returned_unchanged(self):
        state = {
            "lora_unet_blocks_0_x.alpha": object(),
            "lora_unet_blocks_28_x.alpha": object(),
            "lora_unet_blocks_39_x.alpha": object(),
        }
        self.assertIs(self.hook.prepare_lora_for_anima_40(state), state)

    def test_unsupported_or_unknown_lora_raises(self):
        with self.assertRaisesRegex(self.hook.LoraRemapError, r"40\D"):
            self.hook.prepare_lora_for_anima_40(
                {"lora_unet_blocks_40_x.alpha": object()}
            )
        with self.assertRaisesRegex(self.hook.LoraRemapError, r"blocks_0.*blocks_39"):
            self.hook.prepare_lora_for_anima_40({"metadata.only": object()})

    def test_standard_hook_remaps_and_preserves_all_arguments(self):
        calls = []

        def original(
            model,
            clip,
            lora,
            strength_model,
            strength_clip,
            lora_metadata=None,
            *,
            extra=None,
        ):
            calls.append(
                (
                    model,
                    clip,
                    lora,
                    strength_model,
                    strength_clip,
                    lora_metadata,
                    extra,
                )
            )
            return "model-result", "clip-result"

        self.comfy_sd.load_lora_for_models = original
        self.assertTrue(self.hook.install_global_lora_hooks())
        wrapped = self.comfy_sd.load_lora_for_models
        model = self.make_model()
        clip = object()
        metadata = {"filename": "automatic.safetensors"}
        result = wrapped(
            model,
            clip,
            {"lora_unet_blocks_2_x.alpha": object()},
            0.75,
            0.25,
            metadata,
            extra="kept",
        )

        self.assertEqual(result, ("model-result", "clip-result"))
        self.assertEqual(len(calls), 1)
        self.assertIn("lora_unet_blocks_3_x.alpha", calls[0][2])
        self.assertEqual(calls[0][3:], (0.75, 0.25, metadata, "kept"))

    def test_bypass_hook_remaps_and_keyword_call_is_supported(self):
        calls = []

        def original(model, clip, lora, strength_model, strength_clip, **kwargs):
            calls.append((model, clip, lora, strength_model, strength_clip, kwargs))
            return model, clip

        self.comfy_sd.load_bypass_lora_for_models = original
        self.hook.install_global_lora_hooks()
        model = self.make_model()
        self.comfy_sd.load_bypass_lora_for_models(
            model=model,
            clip=None,
            lora={"lora_unet_blocks_14_x.alpha": object()},
            strength_model=1.0,
            strength_clip=0.0,
            custom="kept",
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("lora_unet_blocks_20_x.alpha", calls[0][2])
        self.assertEqual(calls[0][5], {"custom": "kept"})

    def test_hook_is_idempotent_and_non_target_model_is_untouched(self):
        calls = []
        state = {"not.anima": object()}

        def original(model, clip, lora, strength_model, strength_clip, **kwargs):
            calls.append(lora)
            return model, clip

        self.comfy_sd.load_lora_for_models = original
        self.assertTrue(self.hook.install_global_lora_hooks())
        wrapped = self.comfy_sd.load_lora_for_models
        self.assertFalse(self.hook.install_global_lora_hooks())
        self.assertIs(self.comfy_sd.load_lora_for_models, wrapped)

        non_target = self.make_model(image_model="flux")
        wrapped(non_target, None, state, 1.0, 0.0)
        self.assertEqual(calls, [state])

    def test_existing_power_node_calls_original_loader(self):
        calls = []

        def original(model, clip, lora, strength_model, strength_clip, **kwargs):
            calls.append(lora)
            return model, clip

        self.comfy_sd.load_lora_for_models = original
        self.hook.install_global_lora_hooks()
        cached = self.backend.CachedLora(
            signature=self.backend.FileSignature(1, 1),
            state_dict={"lora_unet_blocks_39_x.alpha": object()},
            metadata=None,
        )
        model = self.make_model()
        returned = self.backend.Anima28To40PowerLoraStack._apply_lora(
            model, cached, 1.0
        )
        self.assertIs(returned, model)
        self.assertEqual(calls, [cached.state_dict])


class BackendTests(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend, self.hook, self.comfy_sd, self.comfy_utils = load_backend(self.root)

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
