"""Anima 28 层 LoRA 到 40 层模型的内存键名映射核心。

本模块只处理内存中的 state dict，不读取 manifest，也不会写出转换后的 LoRA 文件。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

OLD_BLOCK_COUNT = 28
NEW_BLOCK_COUNT = 40
INSERTION_POSITIONS = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)


def build_old_to_new_map(
    old_block_count: int = OLD_BLOCK_COUNT,
    new_block_count: int = NEW_BLOCK_COUNT,
    insertion_positions: tuple[int, ...] = INSERTION_POSITIONS,
) -> dict[int, int]:
    """根据插入层位置生成旧层到新层的一一映射。"""
    insertions = set(insertion_positions)
    if len(insertions) != new_block_count - old_block_count:
        raise ValueError("插入层数量与新旧层数差值不一致")
    if any(index < 0 or index >= new_block_count for index in insertions):
        raise ValueError("插入层位置超出新模型层范围")

    old_to_new: dict[int, int] = {}
    old_index = 0
    for new_index in range(new_block_count):
        if new_index in insertions:
            continue
        if old_index >= old_block_count:
            raise ValueError("生成映射时旧层数量溢出")
        old_to_new[old_index] = new_index
        old_index += 1

    if old_index != old_block_count:
        raise ValueError(f"仅映射了 {old_index} 个旧层，预期 {old_block_count} 个")
    return old_to_new


OLD_TO_NEW = build_old_to_new_map()

# 仅识别 Anima 主干层，避免误把 llm_adapter_blocks_* 当作主模型层。
BLOCK_PATTERNS = (
    re.compile(r"(?P<prefix>lora_unet_blocks_)(?P<idx>\d+)(?P<suffix>_)"),
    re.compile(
        r"(?P<prefix>(?:^|[./])(?:diffusion_model|net)[./]blocks[./])"
        r"(?P<idx>\d+)(?P<suffix>[./])"
    ),
)

# ComfyUI 读取 DoRA 幅度向量时使用 `<base>.dora_scale`，而部分 sd-scripts 分叉
# （例如 Anima 训练脚本）会将其导出为 `<base>.dora_magnitude`。两者含义相同，
# 都是逐输出行的幅度向量，因此加载时统一归一化为 ComfyUI 的键名。
DORA_MAGNITUDE_SUFFIX = ".dora_magnitude"
DORA_SCALE_SUFFIX = ".dora_scale"


def normalize_dora_key(key: str) -> str:
    """把非标准 DoRA 幅度键名归一化为 ComfyUI 的 `.dora_scale`。"""
    if key.endswith(DORA_MAGNITUDE_SUFFIX):
        return key[: -len(DORA_MAGNITUDE_SUFFIX)] + DORA_SCALE_SUFFIX
    return key


def denormalize_dora_key(key: str) -> str:
    """反向恢复 :func:`normalize_dora_key`，用于回查原始 state dict。"""
    if key.endswith(DORA_SCALE_SUFFIX):
        return key[: -len(DORA_SCALE_SUFFIX)] + DORA_MAGNITUDE_SUFFIX
    return key


def reshape_dora_scale(value: Any) -> Any:
    """把 1-D DoRA 幅度向量扩展为 `(dim, 1)`。

    部分训练脚本把幅度向量存成 `(dim,)`，而 ComfyUI 的 `weight_decompose`
    会按 `(dim, 1)` 的逐行幅度进行广播；不扩展会触发形状广播错误。
    """
    try:
        if value.dim() == 1:
            return value.unsqueeze(1)
    except (AttributeError, TypeError):
        pass
    return value


class LoraRemapError(ValueError):
    """LoRA 结构不符合 Anima 28 层映射要求。"""


def find_main_block(key: str) -> tuple[re.Match[str] | None, int | None]:
    """返回键中第一个 Anima 主干层匹配及其索引。"""
    for pattern in BLOCK_PATTERNS:
        match = pattern.search(key)
        if match is not None:
            return match, int(match.group("idx"))
    return None, None


def remap_key(
    key: str,
    old_to_new: Mapping[int, int] = OLD_TO_NEW,
) -> tuple[str, int | None, int | None]:
    """重映射单个键；无主干层索引的键保持原样。"""
    match, old_index = find_main_block(key)
    if match is None or old_index is None:
        return normalize_dora_key(key), None, None
    if old_index not in old_to_new:
        raise LoraRemapError(
            f"键 {key!r} 使用了不支持的主干层 {old_index}；仅支持 0-{OLD_BLOCK_COUNT - 1}"
        )

    new_index = old_to_new[old_index]
    new_key = normalize_dora_key(
        f"{key[:match.start('idx')]}{new_index}{key[match.end('idx'):]}"
    )
    return new_key, old_index, new_index


def remap_lora_state_dict(
    state_dict: Mapping[str, Any],
    *,
    source_name: str = "<memory>",
    old_to_new: Mapping[int, int] = OLD_TO_NEW,
) -> dict[str, Any]:
    """严格校验并返回适用于 40 层模型的 LoRA state dict。

    Tensor 对象仅被重新引用，不执行 clone，也不生成新增 12 层的权重。
    """
    remapped: dict[str, Any] = {}
    main_block_key_count = 0
    collisions: list[str] = []

    for key, value in state_dict.items():
        try:
            new_key, old_index, _ = remap_key(key, old_to_new)
        except LoraRemapError as error:
            raise LoraRemapError(f"LoRA {source_name}: {error}") from error

        if old_index is not None:
            main_block_key_count += 1
        if new_key in remapped:
            collisions.append(new_key)
            continue
        remapped[new_key] = (
            reshape_dora_scale(value)
            if key.endswith(DORA_MAGNITUDE_SUFFIX)
            else value
        )

    if main_block_key_count == 0:
        raise LoraRemapError(
            f"LoRA {source_name} 未包含可识别的 Anima 主干 blocks_0 至 blocks_27 权重"
        )
    if collisions:
        preview = "\n  ".join(collisions[:20])
        suffix = "" if len(collisions) <= 20 else f"\n  ...另有 {len(collisions) - 20} 个"
        raise LoraRemapError(
            f"LoRA {source_name} 映射后发生键名冲突：\n  {preview}{suffix}"
        )

    return remapped


__all__ = [
    "BLOCK_PATTERNS",
    "INSERTION_POSITIONS",
    "LoraRemapError",
    "NEW_BLOCK_COUNT",
    "OLD_BLOCK_COUNT",
    "OLD_TO_NEW",
    "build_old_to_new_map",
    "denormalize_dora_key",
    "find_main_block",
    "normalize_dora_key",
    "remap_key",
    "remap_lora_state_dict",
    "reshape_dora_scale",
]
