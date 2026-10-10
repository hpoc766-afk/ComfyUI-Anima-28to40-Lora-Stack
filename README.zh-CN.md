# Anima 28→40 Power LoRA Stack

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个独立的 ComfyUI 自定义节点包，可在内存中将 Anima 28 层 LoRA 的主模型层键映射到 40 层模型。插件同时提供零连接全局自动映射和接近 Power LoRA Loader 的紧凑堆叠节点；两种方式都不会生成或保存转换后的 `.safetensors` 文件。

## 零连接自动映射

安装并重启 ComfyUI 后，插件会自动接管标准 LoRA 应用入口。使用 ComfyUI 原生 LoRA Loader 或调用相同标准 API 的第三方 Loader 时，无需添加本插件节点：

1. 像平常一样加载 40 层 Anima 模型。
2. 像平常一样把任意数量的 LoRA Loader 连接在 MODEL 链路中。
3. 插件确认目标 MODEL 是 40 层 Anima 后，会在 LoRA 应用前自动判断结构。

自动处理规则：

- 支持 `lora_unet_blocks_<n>_...` 与 `diffusion_model.blocks.<n>...` 两种常见 Anima LoRA 键格式；仅含第 0–27 层的 LoRA 按 28 层格式映射到 40 层。
- 含任意 `blocks_28` 至 `blocks_39` 的 LoRA 视为原生 40 层 LoRA，保持原样。
- 含 `blocks_40` 或更高层号，或没有可识别 Anima 主模型层键的 LoRA 会严格报错。
- 非 Anima 模型、不是 40 层的 Anima 模型以及纯 CLIP 调用完全透传。
- 标准和 Bypass LoRA Loader 均支持自动映射。
- 每次加载 LoRA 时一次完成 key 校验与 28→40 映射，发生在编译前。交给 Loader 的是普通 `dict`，前向执行中没有懒 key 视图或映射回调；Tensor 存储仍复用，不执行 clone。DoRA 的幅度键在加载时转换为 `.dora_scale`，一维幅度向量在此阶段调整为 `(dim, 1)` 视图。
- 仅在加载阶段使用最多 4096 项的有界 LRU 缓存保存键字符串分析结果，不缓存 LoRA state dict 或 Tensor。

使用编译工作流时，按 `MODEL 加载 → LoRA Loader / Power Stack → torch.compile 节点 → 采样器` 连接。Power Stack 在源文件大小和修改时间不变时复用已校验的映射字典。切换 LoRA 或改变被编译模型本身，仍可能需要重新编译。

自动兼容范围是调用 `comfy.sd.load_lora_for_models` 或 `comfy.sd.load_bypass_lora_for_models` 的加载节点。直接操作 `ModelPatcher`，或在本插件加载前保存了旧函数引用的第三方节点，不保证自动转换。

> 判定限制：如果原生 40 层 LoRA 只包含 `blocks_0` 至 `blocks_27`，且没有任何 `blocks_28` 至 `blocks_39` 权重，从键结构上无法与 28 层 LoRA 区分，因此会按 28 层格式映射。

## Power LoRA Stack 节点

需要在一个节点内管理多条 LoRA 时，仍可使用现有 `Anima 28→40 Power LoRA Stack`。它保留动态行、搜索、缩放适配、排序、行内删除和 CLIP 原样透传功能，并会主动绕过全局 Hook，避免二次映射。


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

先将 LoRA 文件放入 `ComfyUI/models/loras`，再按需选择以下模式：

### 自动模式——无需本插件节点

1. 加载 40 层 Anima Checkpoint 或扩散模型。
2. 像原来一样使用 ComfyUI 标准 LoRA Loader，并将其连接在 MODEL 链路中。
3. 直接运行工作流；插件会检测目标模型，并自动映射符合条件的 28 层 LoRA。

### Power 堆叠节点

1. 添加 `loaders/Anima > Anima 28→40 Power LoRA Stack`。
2. 连接 `MODEL`；工作流需要透传 CLIP 时再连接 `CLIP`。
3. 点击 `+ Add LoRA`，选择 28 层 Anima LoRA 并设置 MODEL 强度。
4. 按需添加更多条目、调整应用顺序，并将输出 `MODEL` 连接到后续采样节点。

每个 LoRA 行的右键菜单支持启用/禁用、上移、下移和删除。节点顶部还提供全部启用/全部禁用控制。输出 `CLIP` 是输入对象的原样透传。

点击 LoRA 选择区域会打开可搜索选择器。搜索不区分大小写，可匹配文件名和子目录路径，并支持使用 `↑`、`↓`、`Enter`、`Esc` 键盘操作。选择器使用屏幕空间渲染，不会跟随 ComfyUI 画布缩放而变得过大或过小。

## 演示示例

以下图片由本节点生成，并保留了经过清理的 ComfyUI 工作流元数据。下载任意 PNG 后拖入 ComfyUI 画布，即可查看示例工作流。示例引用的模型和 LoRA 文件不包含在仓库中。

| 两条 LoRA 堆叠 | 三条 LoRA 堆叠 |
| --- | --- |
| ![使用两条 Anima LoRA 堆叠生成的示例](examples/anima-power-lora-stack-two-loras.png) | ![使用三条 Anima LoRA 堆叠生成的示例](examples/anima-power-lora-stack-three-loras.png) |

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

## Power 节点严格校验

出现以下情况时，节点会停止执行并给出明确错误：

- LoRA 中不存在可识别的主模型 `blocks_<index>` 键；
- 识别到的源层索引不在 `0–27` 范围内；
- 映射后产生重复的目标键；
- 选中的 LoRA 文件不存在或无法加载；
- 动态 LoRA 条目配置无效。

自动模式仅在可靠确认目标为 40 层 Anima MODEL 时启用；现有 Power 节点仍严格用于 Anima 28 层 LoRA 到 40 层模型映射。

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

测试包含 CPU `torch.compile(fullgraph=True)` 回归：准备完成后禁止再次解析 key，结果与 eager 一致，重复调用只编译一个图。该测试需要 PyTorch，缺少 PyTorch 时跳过；它不测 CUDA/Inductor 性能，也不代表完整 ComfyUI 工作流已验收。

如果系统已安装 Node.js，可检查前端扩展语法：

```bash
node --check web/anima_power_lora_stack.js
```

仓库会明确排除 `.safetensors` 模型文件和生成的 ZIP 压缩包。
