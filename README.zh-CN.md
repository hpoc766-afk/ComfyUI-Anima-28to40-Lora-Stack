# Anima 28→40 Power LoRA Stack

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个独立的 ComfyUI 自定义节点包，可在内存中将 Anima 28 层 LoRA 的主模型层键映射到 40 层模型。节点提供接近 Power LoRA Loader 的紧凑堆叠界面，可按顺序直接应用多条 LoRA，不会生成或保存转换后的 `.safetensors` 文件。

## 功能特点

- 在内存中在线完成 Anima `28 → 40` 层键映射。
- 支持动态添加 LoRA 行，并可启用/禁用、选择文件、调整强度、排序和删除。
- 多条启用的 LoRA 按界面从上到下依次应用。
- 每条 LoRA 只有一个 MODEL 强度，不向 CLIP 应用 LoRA。
- 输入的 `CLIP` 对象原样透传。
- 禁用项、空选择和强度为 `0` 的条目会直接跳过。
- 缓存会在源 LoRA 文件大小或修改时间变化后自动失效。
- 不依赖 rgthree，也不会写出转换后的 LoRA 文件。

## 安装

### Git 安装

在 `ComfyUI/custom_nodes` 目录中克隆仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hpoc766-afk/ComfyUI-Anima-28to40-Lora-Stack.git
```

### ZIP 安装

下载并解压仓库，确保插件目录位于：

```text
ComfyUI/custom_nodes/ComfyUI-Anima-28to40-Lora-Stack
```

安装后重启 ComfyUI。无需额外安装 Python 依赖，也无需安装 rgthree。

## 使用方法

1. 将原始 28 层 Anima LoRA 放入 `ComfyUI/models/loras`。
2. 在工作流中添加 `loaders/Anima > Anima 28→40 Power LoRA Stack`。
3. 连接 `MODEL`；工作流需要 CLIP 时再连接 `CLIP`。
4. 点击 `+ Add LoRA`，选择 LoRA 并设置对应的 MODEL 强度。
5. 按需添加更多条目，并按照实际应用顺序排列。
6. 将输出 `MODEL` 连接到后续采样节点。输出 `CLIP` 是输入对象的原样透传。

每个 LoRA 行的右键菜单支持启用/禁用、上移、下移和删除。节点顶部还提供全部启用/全部禁用控制。

## 映射行为

内置映射会把 28 个源层放到 40 层模型中的对应位置。以下 12 个新增目标层不会复制 LoRA 权重：

```text
2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36
```

映射示例：

```text
0 → 0
2 → 3
14 → 20
27 → 39
```

不能识别为主模型 `blocks_<index>` 层键的其他键会保持原名，但不会向 CLIP 应用任何 LoRA 权重。

`expand_manifest.json` 仅作为映射参考和测试数据保留，节点运行时不会读取该文件。

## 严格校验

出现以下情况时，节点会停止执行并给出明确错误：

- LoRA 中不存在可识别的主模型 `blocks_<index>` 键；
- 识别到的源层索引不在 `0–27` 范围内；
- 映射后产生重复的目标键；
- 选中的 LoRA 文件不存在或无法加载；
- 动态 LoRA 条目配置无效。

本节点仅支持本仓库实现的 Anima 28 层 LoRA 到 40 层模型映射，不用于其他模型架构。

## 输出说明

| 输出 | 行为 |
| --- | --- |
| `MODEL` | 按界面顺序应用所有启用 LoRA 后的模型。 |
| `CLIP` | 输入的 CLIP 原对象，不应用 LoRA。 |

如果没有有效且启用的 LoRA 条目，节点会直接返回原始 `MODEL` 和 `CLIP`。

## 开发与测试

在插件目录中运行单元测试：

```bash
python -m unittest discover -s tests -v
```

如果系统已安装 Node.js，可检查前端扩展语法：

```bash
node --check web/anima_power_lora_stack.js
```

仓库会明确排除 `.safetensors` 模型文件和生成的 ZIP 压缩包。
