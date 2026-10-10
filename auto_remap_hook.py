"""Global ComfyUI hooks for remapping Anima 28-layer LoRAs to 40 layers."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any, Callable

import comfy.sd

from .remap_lora_28_to_40 import (
    DORA_SCALE_SUFFIX,
    LoraRemapError,
    NEW_BLOCK_COUNT,
    OLD_BLOCK_COUNT,
    OLD_TO_NEW,
    find_main_block,
    normalize_dora_key,
    reshape_dora_scale,
)

_LOGGER = logging.getLogger(__name__)
_HOOK_MARKER = "__anima_28_to_40_auto_hook__"
_ORIGINAL_ATTR = "__anima_28_to_40_original__"
_RUNTIME_SOURCE = "<ComfyUI runtime LoRA>"
_KEY_CACHE_SIZE = 4096

_original_load_lora_for_models: Callable[..., Any] | None = None
_original_load_bypass_lora_for_models: Callable[..., Any] | None = None


def _safe_getattr(value: Any, name: str) -> Any | None:
    """Read an attribute exposed by a third-party wrapper without trusting it."""
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _base_model(model: Any) -> Any | None:
    """Return the BaseModel held by a standard ComfyUI ModelPatcher."""
    if model is None:
        return None
    return _safe_getattr(model, "model")


def is_anima_40_model(model: Any) -> bool:
    """Return True only when MODEL can be reliably identified as 40-layer Anima."""
    base_model = _base_model(model)
    if base_model is None:
        return False

    model_config = _safe_getattr(base_model, "model_config")
    unet_config = _safe_getattr(model_config, "unet_config")
    if not isinstance(unet_config, Mapping):
        return False
    if str(unet_config.get("image_model", "")).strip().lower() != "anima":
        return False

    diffusion_model = _safe_getattr(base_model, "diffusion_model")
    blocks = _safe_getattr(diffusion_model, "blocks")
    if blocks is not None:
        try:
            return len(blocks) == NEW_BLOCK_COUNT
        except Exception:
            # Some wrappers expose blocks but do not allow it to be inspected.
            pass

    # Fall back to configuration only when the live block collection is unreadable.
    try:
        return int(unet_config.get("num_blocks")) == NEW_BLOCK_COUNT
    except (TypeError, ValueError):
        return False


def _source_name(metadata: Any) -> str:
    if not isinstance(metadata, Mapping):
        return _RUNTIME_SOURCE
    for key in (
        "filename",
        "path",
        "name",
        "ss_output_name",
        "modelspec.title",
        "modelspec.architecture",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _RUNTIME_SOURCE


@functools.lru_cache(maxsize=_KEY_CACHE_SIZE)
def _analyze_lora_key(key: str) -> tuple[int | None, str]:
    """Return block index and destination key during load-time preparation."""
    match, index = find_main_block(key)
    if match is None or index is None:
        return None, normalize_dora_key(key)

    if index < OLD_BLOCK_COUNT:
        new_index = OLD_TO_NEW[index]
        exposed_key = normalize_dora_key(
            f"{key[:match.start('idx')]}{new_index}{key[match.end('idx'):]}"
        )
    else:
        exposed_key = normalize_dora_key(key)
    return index, exposed_key


def prepare_lora_for_anima_40(
    lora: Mapping[Any, Any],
    *,
    metadata: Any = None,
) -> dict[Any, Any]:
    """Validate and materialize keys once at LoRA load time, before compilation.

    Only dictionary entries are copied; Tensor storage remains shared, including
    DoRA reshape views. Downstream patching and compiled forwards never need
    this plugin's key lookup logic.
    """
    if not isinstance(lora, Mapping):
        raise TypeError(
            f"LoRA state dict 必须是映射，"
            f"实际为 {type(lora).__name__}"
        )

    # Normalize custom mappings at this boundary; never pass lazy key access on.
    if type(lora) is not dict:
        lora = dict(lora)

    source_name = _source_name(metadata)
    found_main_block = False
    native_40_layer = False
    highest_unsupported: int | None = None
    prepared: dict[Any, Any] = {}
    collisions: list[str] = []

    for key, value in lora.items():
        index, exposed_key = (
            _analyze_lora_key(key) if isinstance(key, str) else (None, key)
        )
        if index is not None:
            found_main_block = True
            if index >= NEW_BLOCK_COUNT:
                highest_unsupported = (
                    index
                    if highest_unsupported is None
                    else max(highest_unsupported, index)
                )
            elif index >= OLD_BLOCK_COUNT:
                native_40_layer = True
        if exposed_key in prepared:
            collisions.append(exposed_key)
        else:
            if isinstance(exposed_key, str) and exposed_key.endswith(DORA_SCALE_SUFFIX):
                value = reshape_dora_scale(value)
            prepared[exposed_key] = value

    if not found_main_block:
        raise LoraRemapError(
            f"LoRA {source_name} 未包含可识别的 Anima "
            f"主干 blocks_0 至 blocks_{NEW_BLOCK_COUNT - 1} 权重"
        )
    if highest_unsupported is not None:
        raise LoraRemapError(
            f"LoRA {source_name} 使用了不支持的主干层 {highest_unsupported}；"
            f"40 层 Anima 仅支持 blocks_0 至 "
            f"blocks_{NEW_BLOCK_COUNT - 1}"
        )

    # Any block in 28-39 identifies an already-native 40-layer state dict.
    if native_40_layer:
        return lora
    if collisions:
        preview = "\n  ".join(collisions[:20])
        suffix = (
            ""
            if len(collisions) <= 20
            else f"\n  ...另有 {len(collisions) - 20} 个"
        )
        raise LoraRemapError(
            f"LoRA {source_name} 映射后发生键名冲突：\n  {preview}{suffix}"
        )

    return prepared


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name)


def _replace_lora_argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], lora: Mapping[Any, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > 2:
        changed = list(args)
        changed[2] = lora
        return tuple(changed), kwargs
    changed_kwargs = dict(kwargs)
    changed_kwargs["lora"] = lora
    return args, changed_kwargs


def _wrap_loader(original: Callable[..., Any], *, supports_metadata: bool) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        model = _argument(args, kwargs, 0, "model")
        lora = _argument(args, kwargs, 2, "lora")
        if is_anima_40_model(model):
            metadata = (
                _argument(args, kwargs, 5, "lora_metadata")
                if supports_metadata
                else None
            )
            prepared = prepare_lora_for_anima_40(lora, metadata=metadata)
            args, kwargs = _replace_lora_argument(args, kwargs, prepared)
        return original(*args, **kwargs)

    setattr(wrapped, _HOOK_MARKER, True)
    setattr(wrapped, _ORIGINAL_ATTR, original)
    return wrapped


def _install_one(
    name: str, *, supports_metadata: bool
) -> tuple[Callable[..., Any] | None, bool]:
    current = getattr(comfy.sd, name, None)
    if current is None:
        return None, False
    if getattr(current, _HOOK_MARKER, False):
        return getattr(current, _ORIGINAL_ATTR), False

    original = current
    setattr(comfy.sd, name, _wrap_loader(original, supports_metadata=supports_metadata))
    return original, True


def install_global_lora_hooks() -> bool:
    """Idempotently install standard and bypass LoRA hooks."""
    global _original_load_lora_for_models
    global _original_load_bypass_lora_for_models

    standard, standard_installed = _install_one(
        "load_lora_for_models", supports_metadata=True
    )
    bypass, bypass_installed = _install_one(
        "load_bypass_lora_for_models", supports_metadata=False
    )
    if standard is None:
        raise RuntimeError(
            "当前 ComfyUI 缺少 comfy.sd.load_lora_for_models"
        )

    _original_load_lora_for_models = standard
    _original_load_bypass_lora_for_models = bypass
    installed = standard_installed or bypass_installed
    if installed:
        _LOGGER.info(
            "Anima 28-to-40 global LoRA auto-remap enabled with load-time key validation."
        )
    return installed


def get_original_load_lora_for_models() -> Callable[..., Any]:
    """Return the unwrapped loader for state dicts already remapped by this plugin."""
    if _original_load_lora_for_models is not None:
        return _original_load_lora_for_models
    current = getattr(comfy.sd, "load_lora_for_models")
    return getattr(current, _ORIGINAL_ATTR, current)


def get_original_load_bypass_lora_for_models() -> Callable[..., Any] | None:
    if _original_load_bypass_lora_for_models is not None:
        return _original_load_bypass_lora_for_models
    current = getattr(comfy.sd, "load_bypass_lora_for_models", None)
    return getattr(current, _ORIGINAL_ATTR, current) if current is not None else None


__all__ = [
    "get_original_load_bypass_lora_for_models",
    "get_original_load_lora_for_models",
    "install_global_lora_hooks",
    "is_anima_40_model",
    "prepare_lora_for_anima_40",
]
