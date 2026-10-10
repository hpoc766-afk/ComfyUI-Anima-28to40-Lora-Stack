# 项目交接文档：ComfyUI Anima 28→40 LoRA 自动映射

> 更新时间：2026-10-10
> 目标读者：后续接手开发、验证或发布的 Agent / 开发者

## 1. 当前目标与实现结论

插件现在同时提供两种使用方式：

1. **零连接全局自动模式（本轮新增）**
   - 插件启动时 Hook ComfyUI 标准 LoRA 应用 API。
   - 用户继续按原工作流连接 Checkpoint、LoRA Loader 和 MODEL，无需增加本插件节点。
   - 只有目标 MODEL 被可靠确认是 40 层 Anima 时，才自动处理 LoRA。
2. **Power LoRA Stack 节点（继续保留）**
   - 节点显示名：`Anima 28→40 Power LoRA Stack`。
   - 兼容已有工作流。
   - 保留多 LoRA 动态行、搜索、缩放适配、排序、启停、行内删除和 CLIP 原样透传。

两种方式都不会输出或保存转换后的 `.safetensors` 文件，也不依赖 rgthree。

2026-10-10：自动模式已移除懒 `RemappedLoraView`，在 LoRA 加载阶段一次完成 key 校验和映射，交给原始 Loader 普通 `dict`。未注册模型前向 Hook；全局包装只覆盖 LoRA 加载 API。编译工作流顺序为 `MODEL 加载 → LoRA Loader / Power Stack → torch.compile → 采样器`。

## 2. 仓库、基线与操作边界

本地开发仓库：

```text
E:/zhuanhuan/ComfyUI-Anima-28to40-Lora-Stack
```

2026-10-10 修改的是 ComfyUI 实际安装副本：

```text
E:/sd/ComfyUI-aki-v2/ComfyUI/custom_nodes/ComfyUI-Anima-28to40-Lora-Stack
```

修改前相关文件备份在 `E:/anima-quantization/lora-compile-fix-backup-20261010-194654`。本次按用户后续授权同步到开发仓库并发布到 GitHub，整合了远端 PR #2 的 DoRA 兼容修复；ZIP 本次不重新打包。

GitHub 仓库：

```text
https://github.com/hpoc766-afk/ComfyUI-Anima-28to40-Lora-Stack
```

本次发布基线包含已合并的两个 PR：

```text
3eab7e5 Merge pull request #2 from CheNing233/main
827a2d5 Merge pull request #1 from buxinzi2233/fix/lora-stack-layout
```

保留 PR #1 的节点布局和行内删除行为；保留 PR #2 的 DoRA `.dora_magnitude` → `.dora_scale` 键名归一化与 `(dim, 1)` 幅度视图，并将这些操作固定在 LoRA 加载阶段执行。

仓库外只读验收样本：

```text
E:/zhuanhuan/1999.safetensors
```

禁止修改、删除、提交、打包或上传该样本。

本次用户已明确授权提交和推送到 GitHub。只发布源码、文档和测试；不打 ZIP，不上传权重或只读验收样本。

## 3. 当前工作区改动

相对发布基线 `3eab7e5`，本次改动为：

```text
M  README.md
M  README.zh-CN.md
M  auto_remap_hook.py
M  tests/test_anima_lora_stack.py
M  HANDOFF.md
```

文件职责：

```text
__init__.py                    插件入口，在注册节点前安装全局 Hook
auto_remap_hook.py             模型检测、LoRA 分类和全局 Hook
anima_lora_stack.py            现有 Power LoRA Stack 后端
remap_lora_28_to_40.py         28→40 纯内存键名映射核心
web/anima_power_lora_stack.js  搜索、缩放适配、动态行和 PR 布局
tests/test_anima_lora_stack.py 映射、Hook 和节点兼容测试
README.md                      英文文档
README.zh-CN.md                中文文档
expand_manifest.json           映射参考，运行时不读取
examples/                      演示 PNG
```

## 4. 全局自动映射实现

### 4.1 安装入口

`__init__.py` 在导入节点映射前执行：

```python
from .auto_remap_hook import install_global_lora_hooks

install_global_lora_hooks()
```

插件安装并重启 ComfyUI 后自动启用，不新增开关、配置文件、HTTP API 或必须连接的节点。

### 4.2 Hook 范围

`auto_remap_hook.py` 包装：

```python
comfy.sd.load_lora_for_models
comfy.sd.load_bypass_lora_for_models
```

实现约束：

- 安装幂等，重复导入不会叠加包装。
- 保存两个原始 Loader 引用。
- 包装函数只做 MODEL 判断和 LoRA state dict 替换。
- 其余位置参数、关键字参数、CLIP、强度、metadata 和返回值均交给原函数。
- 旧版 ComfyUI 没有 `load_bypass_lora_for_models` 时会跳过 Bypass Hook。
- 标准 `load_lora_for_models` 缺失时明确报启动错误。
- 安装成功只输出一次 INFO 日志，不逐 Tensor 输出日志。

### 4.3 40 层 Anima MODEL 检测

`is_anima_40_model(model)` 仅在以下条件都成立时返回 `True`：

1. 存在 MODEL，并可取得 `model.model`。
2. `base_model.model_config.unet_config` 是映射对象。
3. `unet_config["image_model"]` 忽略大小写和首尾空白后为 `anima`。
4. 优先读取 `len(base_model.diffusion_model.blocks) == 40`。
5. 只有实际 `blocks` 不可读取或无法取长度时，才回退检查 `unet_config["num_blocks"] == 40`。

无法可靠判断时保守返回 `False`，包括非 Anima、非 40 层、`model=None`、未知对象和异常属性包装器。此时调用原样透传，不扫描 LoRA。

### 4.4 LoRA 分类规则

确认目标是 40 层 Anima 后，`prepare_lora_for_anima_40()` 扫描可识别的主模型层键。当前兼容 `lora_unet_blocks_<n>_...` 和 `diffusion_model.blocks.<n>...` 两种格式：

- 所有索引均在 `0–27`：加载时在同一轮扫描中完成分类、碰撞校验和映射，返回普通 `dict`，后续访问不再解析 key。
- 出现任意 `28–39`：视为原生 40 层 LoRA，普通输入字典保持原对象；自定义 Mapping 在加载边界物化为普通字典。
- 出现 `40` 或更高：严格报错。
- 没有可识别的 Anima 主模型层键：严格报错。
- 合法混合 `0–39`：按原生 40 层处理。
- 非主层键保持原名；DoRA 幅度键按 PR #2 规则归一化为 `.dora_scale`。普通 Tensor/值对象直接复用，一维 DoRA 幅度在加载时转换为 `(dim, 1)` 视图，不执行 `clone()`。
- 键名分析仅在加载阶段使用最多 4096 项的有界 LRU；缓存只持有字符串和层号，不持有 state dict 或 Tensor。
- 映射时严格检查目标键碰撞，并兼容非规范的前导零层号键。
- 不在前向中校验，不使用 `torch.compiler.disable` 把前向切回 eager；切换 LoRA 或模型本身仍可能触发合理的重新编译。

错误来源名称优先从 LoRA metadata 的以下字段读取：

```text
filename
path
name
ss_output_name
modelspec.title
modelspec.architecture
```

没有名称时使用：

```text
<ComfyUI runtime LoRA>
```

### 4.5 已知判定限制

如果一个原生 40 层 LoRA 只包含 `blocks_0` 至 `blocks_27`，完全没有 `blocks_28` 至 `blocks_39`，则仅凭键结构无法与 28 层 LoRA 区分。当前会按用户确认的规则将其作为 28 层 LoRA 映射。

此限制已写入中英文 README。没有可靠外部标识时，不要增加架构猜测。

## 5. 与 Power LoRA Stack 的协作

Power 节点已经在 `_load_remapped_lora()` 中主动完成 28→40 映射。映射后的结果可能是稀疏 40 层字典，并不一定包含 `28–39`。如果再次进入全局 Hook，可能被误判为 28 层并发生二次映射。

因此 `anima_lora_stack.py` 的 `_apply_lora()` 改为调用：

```python
get_original_load_lora_for_models()
```

而不是调用已经包装的 `comfy.sd.load_lora_for_models`。

必须保持以下兼容行为：

- 显示名：`Anima 28→40 Power LoRA Stack`。
- Python 类名：`Anima28To40PowerLoraStack`。
- 分类：`loaders/Anima`。
- 原有工作流序列化格式不变。
- 多 LoRA 应用顺序不变。
- LoRA 只应用 MODEL。
- CLIP 输入对象原样透传。
- 搜索选择器和屏幕空间缩放渲染不变。
- PR 增加的行内删除和布局行为不变。

## 6. 自动模式兼容边界

保证覆盖调用以下 API 的原生和第三方加载节点：

```python
comfy.sd.load_lora_for_models
comfy.sd.load_bypass_lora_for_models
```

不承诺覆盖：

1. 绕过上述 API、直接操作 `ModelPatcher` 的节点。
2. 在本插件 Hook 安装前通过 `from comfy.sd import load_lora_for_models` 固定保存旧函数引用的第三方节点。
3. 自行实现另一套 LoRA 解析或应用流程的节点。

非目标模型完全透传，避免影响其他模型架构。

## 7. 映射规则

源层 28，目标层 40。新增且不复制 LoRA 权重的目标层：

```text
2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36
```

典型映射：

```text
0  → 0
2  → 3
14 → 20
27 → 39
```

`expand_manifest.json` 只作为映射依据和测试参考，运行时不读取。

## 8. 测试覆盖与命令

测试文件：

```text
tests/test_anima_lora_stack.py
```

当前共有 31 项测试，覆盖：

- 完整 28→40 映射。
- 12 个插入层不产生复制权重。
- 非主层键原样保留。
- 非法层号、无有效主层键和映射碰撞严格报错。
- 40 层 Anima MODEL 检测、配置回退和异常包装器保守处理。
- 非 Anima、非 40 层、未知 MODEL、`model=None` 透传。
- 28 层 LoRA 自动映射并保持值对象身份。
- 普通映射字典复用 Tensor，准备后访问不再调用 key 分析或源 Mapping。
- DoRA 键名归一化、幅度视图只在准备时生成，DoRA 别名冲突被拦截。
- 非法 key 在原始 Loader 调用前报错。
- CPU `torch.compile(fullgraph=True)` 回归：禁止编译阶段模型/key 校验，结果与 eager 一致，两次输入只编译一次。
- 键名 LRU 容量固定，且不缓存 state dict/Tensor。
- 原生 40 层 LoRA 保持同一 state dict 对象。
- 标准与 Bypass API 只调用一次原函数。
- 强度、CLIP、metadata、额外位置参数和关键字参数透传。
- Hook 重复安装幂等。
- 非目标模型不扫描 LoRA。
- Power 节点调用原始 Loader，避免二次映射。
- 禁用项、空项、零强度项跳过。
- 多 LoRA 按数字后缀顺序应用。
- CLIP 原样透传。
- 文件变化后缓存失效；不变时只加载、映射校验一次。

最终验证命令：

```powershell
python -m unittest discover -s tests -v
node --check "web/anima_power_lora_stack.js"
python -m compileall .
git -c safe.directory="E:/sd/ComfyUI-aki-v2/ComfyUI/custom_nodes/ComfyUI-Anima-28to40-Lora-Stack" `
  -C "E:/sd/ComfyUI-aki-v2/ComfyUI/custom_nodes/ComfyUI-Anima-28to40-Lora-Stack" diff --check
```

2026-10-10：合入远端 DoRA 修复后，31 项测试全部通过（使用 ComfyUI 自带 PyTorch 环境，fullgraph 测试未跳过）。fullgraph 回归使用 CPU 计数后端，覆盖预先准备的低秩权重和 DoRA 幅度视图，没有验收完整 ComfyUI CUDA/Inductor 生成工作流。

## 9. `1999.safetensors` 只读验收

本轮只读验收结果：

- 源 Tensor 数量：840。
- 映射后 Tensor 数量：840。
- Tensor 对象身份保持不变。
- 新增目标层没有生成权重。
- 仓库内映射前后均没有 `.safetensors` 文件。
- 没有生成转换后的 LoRA 文件。

任何后续样本验收都必须保持只读。

## 10. 文档状态

已经更新：

- `README.md`
- `README.zh-CN.md`
- `HANDOFF.md`

中英文 README 已说明：

- 零连接自动模式。
- 原生 40 层 LoRA 透传。
- 28/40 层分类规则。
- 40+ 与未知结构严格错误。
- 标准 API 兼容边界。
- 稀疏原生 40 层 LoRA 的判定限制。
- 现有 Power 节点继续可用且避免二次映射。

## 11. 操作约束

- 始终使用简体中文回复用户。
- 先读后写，采用最小、可验证改动。
- 遵循 KISS、DRY、YAGNI、SOLID。
- 路径参数使用双引号。
- 不依赖 rgthree。
- 不恢复离线转换文件输出能力。
- 不修改、删除或上传 `1999.safetensors`。
- 不全局安装或升级依赖。
- 不执行 `git reset --hard`。
- 本轮不要执行 `git commit`、`git push` 或重新打 ZIP。
- 后续如需提交、推送、删除或批量移动，必须重新取得用户明确确认。

## 12. 建议接手流程

1. 阅读本文件、`README.zh-CN.md`、`auto_remap_hook.py` 和相关测试。
2. 检查 Git 状态与完整 diff，确认没有覆盖用户或 PR 的未提交修改。
3. 执行第 8 节的全部验证命令。
4. 对 `1999.safetensors` 做只读映射验收。
5. 在真实 ComfyUI 中手工验证标准 LoRA Loader、Bypass Loader 和旧 Power 节点。
6. 只有用户再次明确授权后，才提交、推送或重新发布 ZIP。
