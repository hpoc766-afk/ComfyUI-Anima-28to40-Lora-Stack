# Anima 28→40 Power LoRA Stack

[English](README.md) | [简体中文](README.zh-CN.md)

A standalone custom node for ComfyUI that remaps Anima 28-layer LoRA backbone keys to a 40-layer model entirely in memory. Multiple LoRAs can be arranged and applied in order through a compact Power LoRA Loader-style interface, without writing converted `.safetensors` files.

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

1. Place the original 28-layer Anima LoRA files under `ComfyUI/models/loras`.
2. Add `loaders/Anima > Anima 28→40 Power LoRA Stack` to the workflow.
3. Connect a `MODEL`; connect `CLIP` only when the workflow needs it.
4. Click `+ Add LoRA`, choose a LoRA, and set its MODEL strength.
5. Add more rows as needed and arrange them in the order they should be applied.
6. Connect the output `MODEL` to the downstream sampling workflow. The output `CLIP` is the original input object.

Each row's context menu supports enable/disable, move up, move down, and delete. The node also provides a top-level control for enabling or disabling all rows.

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

## Strict Validation

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

Check the frontend extension syntax with Node.js when available:

```bash
node --check web/anima_power_lora_stack.js
```

The repository intentionally excludes `.safetensors` model files and generated ZIP archives.
