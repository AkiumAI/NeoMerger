[README.md](https://github.com/user-attachments/files/26316896/README.md)
# NeoMerger

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Forge Neo](https://img.shields.io/badge/Forge%20Neo-compatible-green)
![A1111](https://img.shields.io/badge/A1111-compatible-green)

**Advanced model merging extension for Forge Neo, Forge, reForge, and AUTOMATIC1111.**

Merge SDXL, Pony, Illustrious and SD 1.x checkpoints with block-level precision — no command line required.

---

## Features

- **Block Merge** — blend two checkpoints at the block level with Easy (semantic) or Normal (per-block) mode
- **LoRA Bake-in** — permanently fuse a LoRA into a checkpoint
- **LoRA Merge** — combine two LoRA files into one
- **Model Inspector** — read architecture, metadata, merge recipes, and training info from any `.safetensors` or `.ckpt` file
- **Precision control** — save in fp16, bf16, fp32, or fp8
- **VAE swap** — replace the checkpoint VAE before saving
- **Presets** — save and reload your merge configurations
- **Metadata embedding** — NeoMerger writes the merge recipe inside the output file so you can always trace what was merged

---

## Compatibility

| WebUI | Supported |
|---|---|
| [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) | ✅ |
| [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) | ✅ |
| [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) | ✅ |
| [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | ✅ |

| Architecture | Block Merge | LoRA Tools |
|---|---|---|
| SDXL / Pony / Illustrious | ✅ Full support | ✅ |
| SD 1.x | ⚠️ Works, block labels are SDXL-based | ✅ |
| Flux | ❌ Not supported | ⚠️ Partial |

---

## Installation

1. Download the latest release zip
2. Extract and place the `neo-merger` folder inside your `extensions/` directory:

```
stable-diffusion-webui-forge/
└── extensions/
    └── neo-merger/          ← here
        ├── install.py
        ├── README.md
        └── scripts/
            └── neo_merger.py
```

3. Restart your WebUI
4. The **NeoMerger** tab will appear in the interface

---

## Usage

### Block Merge

Select two checkpoints (Model A and Model B), choose a mode, and click **Run Merge**.

**Easy mode** — six semantic sliders control what each model contributes:

| Slider | Controls |
|---|---|
| Style / Colors | Artistic look, color palette, brushwork |
| Anatomy / Proportions | Body structure, posture, correctness |
| Face Details | Face quality, eyes, expressions |
| Background / Scene | Environment, perspective, composition |
| NSFW / Mature Content | Mature content characteristics from Model B |
| Detail / Sharpness | Fine detail, texture, sharpness |

> **Experimental sliders** (Saturation, Contrast, Brightness, Sharpness, Lights & Darkness) are available in a collapsed section. Results vary between models and are not guaranteed.

**Normal mode** — direct control over all 20 SDXL blocks (BASE, IN00–IN08, MID, OUT00–OUT08).

### LoRA Bake-in

Permanently fuse a LoRA into a checkpoint at a chosen strength. Advanced mode lets you control the strength per semantic category.

### LoRA Merge

Combine two LoRA files with a blend slider. Handles different ranks automatically.

### Model Inspector

Load any model or LoRA to inspect:
- File size, SHA256 hash, detected architecture
- Total keys and parameters
- Embedded metadata (NeoMerger recipe, Supermerger recipe, LoRA training info, trigger words)

### Advanced Options

Available in all merge tabs:
- **Precision** — fp16 (default), bf16, fp32, fp8 (experimental)
- **VAE swap** — replace the VAE before saving
- **Save metadata** — embed the merge recipe in the output file

---

## System Requirements

- **RAM** — at least 16 GB recommended for SDXL merges (two SDXL models = ~12 GB RAM during merge)
- **Disk** — ~7 GB free per merge output
- **GPU** — not required for merging (CPU only)

---

## Presets

Presets are saved as JSON files in `extensions/neo-merger/presets/`. You can back them up, share them, or commit them to your own fork.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

When reporting a bug, please include:
- Your WebUI version and type (Forge Neo, A1111, etc.)
- The error message from the terminal
- The model architectures you were merging

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — see [LICENSE](LICENSE)

You are free to use, share, and modify this project for **non-commercial purposes** with attribution.
Commercial use is not permitted without explicit written permission from the author.
