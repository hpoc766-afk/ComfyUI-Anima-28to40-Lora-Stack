"""ComfyUI 后端节点：在线重映射并堆叠 Anima 28 层 LoRA。"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.sd
import comfy.utils
import folder_paths

from .remap_lora_28_to_40 import LoraRemapError, remap_lora_state_dict


class AnyType(str):
    """允许 ComfyUI 将任意动态 lora_* 输入传给节点。"""

    def __ne__(self, other: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """同时声明固定可选输入并接收任意动态输入。"""

    def __init__(self, dynamic_type: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(data or {})
        self.dynamic_type = dynamic_type
        self.data = data or {}

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        return (self.dynamic_type,)

    def __contains__(self, key: object) -> bool:
        return True


ANY_TYPE = AnyType("*")
LORA_INPUT_PATTERN = re.compile(r"^lora_(?P<index>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class FileSignature:
    """用于判断磁盘 LoRA 是否变化。"""

    size: int
    modified_ns: int


@dataclass
class CachedLora:
    """节点实例内的已映射 LoRA 缓存。"""

    signature: FileSignature
    state_dict: dict[str, Any]
    metadata: dict[str, str] | None


def _lora_catalog() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    return names or [""]


def _file_signature(path: Path) -> FileSignature:
    stat = path.stat()
    return FileSignature(size=stat.st_size, modified_ns=stat.st_mtime_ns)


def _parse_lora_inputs(kwargs: dict[str, Any]) -> list[tuple[int, str, float]]:
    """严格校验动态配置并返回启用且强度非零的条目。"""
    parsed: list[tuple[int, str, float]] = []
    for key, value in kwargs.items():
        if not key.lower().startswith("lora_"):
            continue

        match = LORA_INPUT_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"动态 LoRA 输入名 {key!r} 无效，应为 lora_1、lora_2 等格式")
        if not isinstance(value, dict):
            raise ValueError(f"{key} 配置必须是对象，实际为 {type(value).__name__}")

        missing = {field for field in ("on", "lora", "strength") if field not in value}
        if missing:
            raise ValueError(f"{key} 缺少字段：{', '.join(sorted(missing))}")
        if not isinstance(value["on"], bool):
            raise ValueError(f"{key}.on 必须是布尔值")
        if not isinstance(value["lora"], str):
            raise ValueError(f"{key}.lora 必须是字符串")
        strength = value["strength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise ValueError(f"{key}.strength 必须是数字")
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError(f"{key}.strength 必须是有限数字")

        lora_name = value["lora"].strip()
        if value["on"] and lora_name and strength != 0:
            parsed.append((int(match.group("index")), lora_name, strength))

    parsed.sort(key=lambda item: item[0])
    return parsed


class Anima28To40PowerLoraStack:
    """动态堆叠多个 28 层 Anima LoRA，并在线应用到 40 层 MODEL。"""

    CATEGORY = "loaders/Anima"
    FUNCTION = "load_loras"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    DESCRIPTION = "在内存中将 Anima 28 层 LoRA 映射到 40 层模型，不写出转换文件。"

    def __init__(self) -> None:
        self._cache: dict[Path, CachedLora] = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        catalog = _lora_catalog()
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": FlexibleOptionalInputType(
                ANY_TYPE,
                {
                    "clip": ("CLIP",),
                    # 前端隐藏此 Combo，仅借助它获得 ComfyUI 的 LoRA 文件列表。
                    "_lora_catalog": (catalog, {"default": catalog[0]}),
                },
            ),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs: Any) -> bool:
        # 动态 lora_* 是自定义字典值，结构校验在执行阶段给出更清晰的错误。
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        """把 LoRA 文件状态纳入 ComfyUI 执行缓存指纹。"""
        fingerprints: list[str] = []
        try:
            entries = _parse_lora_inputs(kwargs)
        except ValueError as error:
            return f"invalid:{error}"
        for index, lora_name, strength in entries:
            try:
                path = Path(folder_paths.get_full_path_or_raise("loras", lora_name)).resolve()
                signature = _file_signature(path)
                fingerprints.append(
                    f"{index}:{path}:{signature.size}:{signature.modified_ns}:{strength}"
                )
            except (OSError, FileNotFoundError, ValueError) as error:
                fingerprints.append(f"{index}:{lora_name}:missing:{error}")
        return "|".join(fingerprints)

    def _load_remapped_lora(self, lora_name: str) -> tuple[Path, CachedLora]:
        try:
            path = Path(folder_paths.get_full_path_or_raise("loras", lora_name)).resolve()
            signature = _file_signature(path)
        except Exception as error:
            raise FileNotFoundError(f"无法定位 LoRA {lora_name!r}: {error}") from error

        cached = self._cache.get(path)
        if cached is not None and cached.signature == signature:
            return path, cached

        try:
            loaded = self._load_torch_file(path)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                state_dict, metadata = loaded
            else:
                state_dict, metadata = loaded, None
            if not isinstance(state_dict, dict):
                raise TypeError(f"加载结果不是 state dict，而是 {type(state_dict).__name__}")
            remapped = remap_lora_state_dict(state_dict, source_name=lora_name)
        except (LoraRemapError, OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"加载或映射 LoRA {lora_name!r} 失败：{error}") from error

        cached = CachedLora(signature=signature, state_dict=remapped, metadata=metadata)
        self._cache[path] = cached
        return path, cached

    @staticmethod
    def _load_torch_file(path: Path) -> Any:
        """兼容支持及不支持 return_metadata 的 ComfyUI 版本。"""
        try:
            return comfy.utils.load_torch_file(
                str(path), safe_load=True, return_metadata=True
            )
        except TypeError as error:
            if "return_metadata" not in str(error):
                raise
            return comfy.utils.load_torch_file(str(path), safe_load=True)

    @staticmethod
    def _apply_lora(model: Any, cached: CachedLora, strength: float) -> Any:
        loader = comfy.sd.load_lora_for_models
        parameters = inspect.signature(loader).parameters
        if "lora_metadata" in parameters:
            model, _ = loader(
                model,
                None,
                cached.state_dict,
                strength,
                0.0,
                lora_metadata=cached.metadata,
            )
        else:
            model, _ = loader(model, None, cached.state_dict, strength, 0.0)
        return model

    def load_loras(
        self,
        model: Any,
        clip: Any = None,
        _lora_catalog: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        entries = _parse_lora_inputs(kwargs)
        if not entries:
            self._cache.clear()
            return model, clip

        active_paths: set[Path] = set()
        result_model = model
        try:
            for _, lora_name, strength in entries:
                path, cached = self._load_remapped_lora(lora_name)
                active_paths.add(path)
                result_model = self._apply_lora(result_model, cached, strength)
        finally:
            # 只保留本次实际使用的文件，避免节点长期使用时缓存无限增长。
            self._cache = {
                path: cached for path, cached in self._cache.items() if path in active_paths
            }

        # Anima LoRA 仅应用 MODEL；CLIP 明确原样透传。
        return result_model, clip


NODE_CLASS_MAPPINGS = {
    "Anima28To40PowerLoraStack": Anima28To40PowerLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima28To40PowerLoraStack": "Anima 28→40 Power LoRA Stack",
}


__all__ = [
    "Anima28To40PowerLoraStack",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
