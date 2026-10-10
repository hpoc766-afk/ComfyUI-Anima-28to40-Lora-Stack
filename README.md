# Anima 28→40 Power LoRA Stack

[English](README.md) | [简体中文](README.zh-CN.md)

A standalone ComfyUI extension that remaps Anima 28-layer LoRA backbone keys to a 40-layer model entirely in memory. It provides both zero-connection global auto-remapping and a compact Power LoRA Loader-style stack node, without writing converted `.safetensors` files.

## Zero-Connection Automatic Remapping

After installation and a ComfyUI restart, the extension hooks the standard LoRA application APIs. No plugin node is required when using ComfyUI's native LoRA Loader or third-party loaders that call the same APIs:

1. Load a 40-layer Anima model normally.
2. Connect any number of LoRA loaders in the MODEL chain as usual.
3. Before each LoRA is applied, the extension verifies the target MODEL and classifies the LoRA layout.

Automatic behavior:

- Both common Anima LoRA key formats, `lora_unet_blocks_<n>_...` and `diffusion_model.blocks.<n>...`, are supported; LoRAs containing only layers 0–27 are treated as 28-layer data and remapped.
- A LoRA containing any `blocks_28` through `blocks_39` key is treated as native 40-layer data and passed through unchanged.
- `blocks_40` or higher, or a LoRA with no recognizable Anima backbone keys, causes a strict error.
- Non-Anima models, non-40-layer Anima models, and CLIP-only calls pass through unchanged.
- Both standard and bypass LoRA loading APIs are covered.
- Key validation and 28→40 remapping finish once during each LoRA load, before compilation. The loader receives a plain `dict`; no lazy key view or remapping callback remains in forward execution. Tensor storage is shared without cloning; DoRA magnitude keys are normalized to `.dora_scale`, with one-dimensional magnitudes reshaped to `(dim, 1)` during loading.
- A bounded 4,096-entry LRU caches key-string analysis only at load time; it never caches a LoRA state dict or Tensor.

For compiled workflows, connect `MODEL loader → LoRA loader(s) / Power Stack → torch.compile node → sampler`. The Power Stack reuses its validated mapped dictionary while the source file size and modification time remain unchanged. Changing a LoRA or the compiled model can still require recompilation.

Automatic compatibility is limited to loaders that call `comfy.sd.load_lora_for_models` or `comfy.sd.load_bypass_lora_for_models`. Nodes that patch `ModelPatcher` directly, or capture the original function before this extension loads, are not guaranteed to be intercepted.

> Detection limitation: a native 40-layer LoRA that contains only `blocks_0` through `blocks_27` and no `blocks_28` through `blocks_39` keys cannot be distinguished from a 28-layer LoRA by key structure, so it will be remapped as 28-layer data.

## Power LoRA Stack Node

The existing `Anima 28→40 Power LoRA Stack` remains available when you want to manage multiple LoRAs in one node. Dynamic rows, search, zoom-aware rendering, sorting, inline deletion, and CLIP pass-through are preserved. The node deliberately calls the original ComfyUI loader after its own remap to prevent double remapping by the global hook.

## Features

- Online, in-memory Anima `28 → 40` layer-key remapping.
- Dynamic LoRA rows with enable/disable, selection, strength, reordering, and removal controls.
- Applies enabled LoRAs sequentially from top to bottom.
- Uses one MODEL strength per LoRA; LoRAs are not applied to CLIP.
- Passes the input `CLIP` object through unchanged.
- Skips disabled rows, empty selections, and rows with strength `0`.
- Invalidates the mapped-LoRA cache when the source file size or modification time changes.
- Does not depend on rgthree and does not write converted LoRA files.

## Installation

### Git

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hpoc766-afk/ComfyUI-Anima-28to40-Lora-Stack.git
```

### ZIP

Download and extract the repository so the plugin directory is located at:

```text
ComfyUI/custom_nodes/ComfyUI-Anima-28to40-Lora-Stack
```

Restart ComfyUI after installation. No additional Python packages or rgthree installation are required.

## Usage

Place the LoRA files under `ComfyUI/models/loras`, then choose either mode:

### Automatic mode - no plugin node

1. Load a 40-layer Anima checkpoint or diffusion model.
2. Use ComfyUI's standard LoRA Loader nodes exactly as before and connect them in the MODEL chain.
3. Run the workflow. The extension detects the target model and remaps eligible 28-layer LoRAs automatically.

### Power stack node

1. Add `loaders/Anima > Anima 28→40 Power LoRA Stack`.
2. Connect `MODEL`; connect `CLIP` only when the workflow needs pass-through.
3. Click `+ Add LoRA`, choose a 28-layer Anima LoRA, and set its MODEL strength.
4. Add more rows as needed, arrange their application order, and connect the output `MODEL` to the sampler path.

Each LoRA row has a context menu for enable/disable, move up, move down, and remove. The node also has a global enable/disable control at the top. Its output `CLIP` is the original input object.

Clicking the LoRA selector opens a searchable picker. Search is case-insensitive, matches both file names and subdirectory paths, and supports `↑`, `↓`, `Enter`, and `Esc`. The picker is rendered in screen space, so it stays usable instead of scaling with the ComfyUI canvas zoom.

## Examples

The following images were generated with the node and retain sanitized ComfyUI workflow metadata. You can download either PNG and drag it onto the ComfyUI canvas to inspect the example workflow. The referenced model and LoRA files are not included.

| Two-LoRA stack | Three-LoRA stack |
| --- | --- |
| ![Example generated with two stacked Anima LoRAs](examples/anima-power-lora-stack-two-loras.png) | ![Example generated with three stacked Anima LoRAs](examples/anima-power-lora-stack-three-loras.png) |

## Remapping Behavior

The built-in mapping moves the 28 source blocks into their corresponding positions in the 40-layer model. The 12 inserted destination layers do not receive copied LoRA weights:

```text
2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36
```

Examples:

```text
0 → 0
2 → 3
14 → 20
27 → 39
```

Keys that are not recognized as main `blocks_<index>` keys are preserved under their original names, but nothing is applied to CLIP.

`expand_manifest.json` is included as a mapping reference and test fixture. The node does not read it at runtime.

## Power Node Strict Validation

Execution stops with a clear error when:

- the LoRA contains no recognizable main-model `blocks_<index>` key;
- a recognized source block index is outside `0–27`;
- remapping produces duplicate destination keys;
- a selected LoRA file is missing or cannot be loaded; or
- a dynamic LoRA row contains an invalid configuration.

This node only supports the Anima 28-layer LoRA to 40-layer model mapping implemented in this repository.

## Outputs

| Output | Behavior |
| --- | --- |
| `MODEL` | The input model with all enabled LoRAs applied in row order. |
| `CLIP` | The exact input CLIP object, passed through without LoRA application. |

If no valid LoRA row is enabled, the original `MODEL` and `CLIP` are returned unchanged.

## Development and Tests

Run the unit tests from the plugin directory:

```bash
python -m unittest discover -s tests -v
```

The tests include a CPU `torch.compile(fullgraph=True)` regression using a counting backend: key analysis is forbidden after preparation, results match eager execution, and repeated calls use one graph. PyTorch is required to run this test; it is skipped when PyTorch is unavailable. This does not benchmark CUDA/Inductor or certify a complete ComfyUI workflow.

Check the frontend extension syntax with Node.js when available:

```bash
node --check web/anima_power_lora_stack.js
```

The repository intentionally excludes `.safetensors` model files and generated ZIP archives.
