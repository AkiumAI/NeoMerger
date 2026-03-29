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

<img width="1851" height="919" alt="neomergerui" src="https://github.com/user-attachments/assets/fe312e74-b165-4e5a-8114-0439d43d4129" />

---

### Merge tab

<img width="1242" height="566" alt="neomergereasyblocks" src="https://github.com/user-attachments/assets/6875598d-94f6-4f44-8f66-185cfa9fc335" />

Select two checkpoints (Model A and Model B), pick a merge method, and click **Run Merge**.

**Merge methods** — Weighted Sum, SLERP, Add Difference, TIES, and DARE. Add Difference requires a third model (C).

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

<img width="1217" height="925" alt="neomergerblocksui" src="https://github.com/user-attachments/assets/58311866-11a5-4666-a0fb-6a63fb588855" />

**Normal mode** — direct control over all 20 SDXL blocks (BASE, IN00–IN08, MID, OUT00–OUT08). Toggle the description button to see what each block influences, with warnings for blocks where float values tend to cause issues.

<img width="686" height="841" alt="neomergeranalyzerui" src="https://github.com/user-attachments/assets/fd9398bc-d833-4eeb-b6c6-2d9ac1c04ddc" />


**Block Similarity Analyzer** — before merging, analyze how different the two models are at the block level. Shows a color-coded chart from blue (nearly identical) to red (very different), with semantic tags describing what each block influences.

---

### LoRA Tools tab

<img width="1834" height="947" alt="neomergerloraui" src="https://github.com/user-attachments/assets/7d81b6ea-d1fe-4758-8c9a-89c6ab995d22" />

**Bake-in** — permanently fuse a LoRA into a checkpoint at a chosen strength. Easy mode uses a single strength slider (0–2). Advanced mode gives you per-category control with the same semantic sliders as the Merge tab.

**LoRA Merge** — combine two LoRA files with a blend slider. Handles different ranks automatically by zero-padding the smaller one.

---

### Inspect tab

<img width="1828" height="925" alt="neomergerinspectui" src="https://github.com/user-attachments/assets/abea4721-32b6-48da-b001-f7f693ca5a7d" />

Load any model or LoRA to inspect:
- File size, SHA256 hash, detected architecture
- Total keys and parameters
- Embedded metadata — NeoMerger recipe, Supermerger recipe, LoRA training info, trigger words

You can also clear all metadata from a file.

---

### Advanced Options

Available in all merge tabs:
- **Precision** — fp16 (default), bf16, fp32, fp8 (experimental)
- **VAE swap** — replace the VAE before saving
- **Save metadata** — embed the merge recipe in the output file

---

## System Requirements

- **RAM** — at least 16 GB recommended for SDXL merges (two SDXL models = ~12 GB RAM during merge)
- **Disk** — ~7 GB free per merge output

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

## Acknowledgements

Big thanks to the beta testers who helped shape NeoMerger:

- [Vetehine](https://civitai.com/user/Vetehine) — testing, feedback and bug reports
- [EarthboundAI](https://civitai.com/user/EarthboundAI) — testing, feedback and bug reports

---

## Author

**AkiumAI** — [github.com/AkiumAI](https://github.com/AkiumAI) · [CivitAI](https://civitai.com/user/Akium)


---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — see [LICENSE](LICENSE)

You are free to use, share, and modify this project for **non-commercial purposes** with attribution.
Commercial use is not permitted without explicit written permission from the author.

> This extension is provided as-is, without warranty of any kind. The author is not responsible
> for any damages or data loss resulting from its use. Always keep backups of your original models
> before merging.
