"""Anima 28→40 Power LoRA Stack 自定义节点入口。"""

from .auto_remap_hook import install_global_lora_hooks

# 在注册节点前启用零连接自动映射；重复导入时安装过程保持幂等。
install_global_lora_hooks()

from .anima_lora_stack import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
