# Anima 28→40 Power LoRA Stack

独立的 ComfyUI 自定义节点。它在内存中把 28 层 Anima LoRA 的主干层键映射到 40 层模型，并按顺序直接应用到 `MODEL`，不会生成转换后的 LoRA 文件。

## 安装

将本目录放入：

```text
ComfyUI/custom_nodes/anima_28_to_40_lora_stack
```

然后重启 ComfyUI。无需安装 rgthree。

## 使用

1. 把原始 28 层 Anima LoRA 放入 `ComfyUI/models/loras`。
2. 添加 `loaders/Anima > Anima 28→40 Power LoRA Stack`。
3. 连接 `MODEL`；如工作流需要，也可连接 `CLIP`。
4. 点击 `+ Add LoRA` 添加条目，选择 LoRA 并设置强度。
5. 输出 `MODEL` 连接到后续采样节点；输出 `CLIP` 为原样透传。

节点只接受主干层号为 `0-27` 的目标 LoRA。检测到 28 以上层号、无可识别主干层或映射键冲突时会停止执行并报告错误。

## 行为说明

- 内置 `28 → 40` 层映射，不在运行时读取 `expand_manifest.json`。
- 新增的 12 个模型层不会复制 LoRA 权重。
- 多条 LoRA 按界面从上到下依次应用。
- 禁用、空文件名、强度为 `0` 的条目会跳过。
- 缓存根据 LoRA 文件大小和修改时间自动失效。
- 不写入或输出新的 `.safetensors` 文件。
