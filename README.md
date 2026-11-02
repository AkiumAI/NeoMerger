# NeoMerger

[![ko-fi](https://img.shields.io/badge/Ko--fi-Support-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/akinak4)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Forge Neo](https://img.shields.io/badge/Forge%20Neo-compatible-green)
![A1111](https://img.shields.io/badge/A1111-compatible-green)

**Advanced model merging extension for Forge Neo, Forge, reForge, and AUTOMATIC1111.**

Merge SDXL, Pony, Illustrious, Anima, Krea 2 and MOSS-Image checkpoints with block-level precision, bake LoRAs into checkpoints, extract LoRAs from finetunes. No command line required.

---

## What's new in 1.5

- **Krea 2 and MOSS-Image** support for merging, block merging and LoRA bake-in *(experimental)*
- **Streaming save**: the output is written to disk tensor by tensor, so even 26 GB models merge on 16 GB of RAM
- **fp8 checkpoints** (including the official scaled-fp8 releases) can be used as merge and bake-in sources
- **New merge methods**: Train Difference and Model Stock
- **Multiple donors**: merge up to four models into Model A in one pass
- **Per-block weights on every method** except Model Stock, including TIES, DARE, Task Arithmetic, Breadcrumbs and DELLA
- **Anima Easy mode**: semantic sliders for Anima's DiT blocks
- **Anima 2.9B / 3.8B**: depth-upscaled Anima models merge correctly with the base 2B and with each other
- **LoRA Extract**: build a LoRA from the difference between two checkpoints
- **Alpha Sweep**: merge a series of alphas and compare them in one grid
- **LoKr and DoRA** LoRAs in bake-in
- Norm clamp, Legacy mode, automatic output names and a RAM estimate before every merge

See the changelog for the full list.

---

## Features

- **Block Merge**: blend checkpoints per block, with semantic sliders (Easy, Anima Easy) or one slider per block (Normal for SDXL, Anima for DiT models)
- **11 merge methods** and up to 4 donor models per merge
- **Merge & Gen**: merge, load the result and generate a preview with your txt2img settings
- **Alpha Sweep**: one merge per alpha value, shown side by side
- **Block Similarity Analyzer**: see how different two models are, block by block
- **LoRA Bake-in**: permanently fuse a LoRA into a checkpoint
- **LoRA Merge**: combine two LoRA files into one
- **LoRA Extract**: turn the difference between two checkpoints into a LoRA
- **Model Inspector**: architecture, metadata, merge recipes and training info of any `.safetensors` or `.ckpt` file
- **Block Probe**: swap one block at a time from Model B into the loaded model and see what it changes, without writing any file
- **Precision control**: save in fp16, bf16, fp32 or fp8
- **VAE swap**, **presets** and **metadata embedding**: every output records the recipe that made it

---

## Compatibility

| WebUI | Supported |
|---|---|
| [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) | ✅ |
| [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) | ✅ |
| [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) | ✅ |
| [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | ✅ |

| Architecture | Merge / Block Merge | LoRA Bake-in | LoRA Extract |
|---|---|---|---|
| SDXL / Pony / Illustrious | ✅ 20 blocks (BASE, IN00–IN08, MID, OUT00–OUT08) | ✅ UNet + text encoders | ✅ |
| Anima (2B, 2.9B, 3.8B) | ✅ 28 / 40 / 52 DiT blocks | ✅ | ✅ |
| Krea 2 | ⚠️ Experimental, 28 DiT blocks | ⚠️ Experimental | ❌ |
| MOSS-Image | ⚠️ Experimental, 42 DiT blocks | ⚠️ Experimental | ❌ |
| SD 1.x | ⚠️ Works, block labels are SDXL-based | ⚠️ UNet only | ⚠️ Untested |
| Flux | ❌ Not supported | ❌ | ❌ |

Krea 2 and MOSS-Image follow Forge Neo's own key layout and LoRA naming, but have not yet been tested on real weights. Please report anything odd.

LoRA Merge works with any two LoRAs that share the same keys.

Quantized sources: fp8 checkpoints are dequantized on read. int8, nvfp4, Nunchaku and bitsandbytes checkpoints are refused, and GGUF files are not supported: merge the bf16 release and quantize the result afterwards.

---

## Installation

1. Download the latest release zip
2. Extract it and place the `neomerger` folder inside your `extensions/` directory:

```
stable-diffusion-webui-forge/
└── extensions/
    └── neomerger/          ← here
        ├── install.py
        ├── README.md
        ├── javascript/
        │   └── neomerger.js
        └── scripts/
            └── neomerger.py
```

3. Restart your WebUI
4. The **NeoMerger** tab will appear in the interface

When updating, replace both `scripts/neomerger.py` and `javascript/neomerger.js`: Merge & Gen, Alpha Sweep and Block Probe need the matching script.

---

## Usage

<img width="1851" height="919" alt="neomergerui" src="https://github.com/user-attachments/assets/fe312e74-b165-4e5a-8114-0439d43d4129" />

---

### Merge tab

<img width="1242" height="566" alt="neomergereasyblocks" src="https://github.com/user-attachments/assets/6875598d-94f6-4f44-8f66-185cfa9fc335" />

Select Model A and Model B, pick a merge method, and click **Run Merge**. The architecture is detected automatically.

**Merge methods**

| Method | What it does |
|---|---|
| Weighted Sum | Plain blend between A and B |
| SLERP / NuSLERP | Spherical blend; NuSLERP is smoother and has an optional row-wise mode |
| Add Difference | Adds (B − C) to A. Needs Model C |
| Train Difference | Like Add Difference, but transfers what B learned on top of C more selectively. Needs Model C |
| Task Arithmetic | Adds B's changes to A. With a negative Alpha it *removes* a style |
| TIES / DARE | Merge the important changes only, and resolve conflicts between donors |
| Breadcrumbs | Ignores both outliers and tiny noise: good on models that were merged many times |
| DELLA | Like DARE, but keeps important changes more often than minor ones |
| Model Stock | Works out its own strength from the angle between the donors. Needs at least two donors |

**More donors**: open *More donors* to add up to three extra models, each with its own weight. Weighted Sum, TIES, DARE, Task Arithmetic, Breadcrumbs, DELLA and Model Stock accept several donors. If the weights add up to more than the method can take, NeoMerger warns you before merging.

**Quick actions**: *Add style* and *Remove style* set up a Breadcrumbs merge in one click.

**Block modes**

- **Easy**: six semantic sliders for SDXL models

| Slider | Controls |
|---|---|
| Style / Colors | Artistic look, color palette, brushwork |
| Anatomy / Proportions | Body structure, posture, correctness |
| Face Details | Face quality, eyes, expressions |
| Background / Scene | Environment, perspective, composition |
| NSFW / Mature Content | Mature content characteristics from Model B |
| Detail / Sharpness | Fine detail, texture, sharpness |

> **Experimental sliders** (Saturation, Contrast, Brightness, Sharpness, Lights & Darkness) are in a collapsed section. Results vary between models.

<img width="1217" height="925" alt="neomergerblocksui" src="https://github.com/user-attachments/assets/58311866-11a5-4666-a0fb-6a63fb588855" />

- **Normal**: one slider for each of the 20 SDXL blocks. The description button shows what each block influences. *Convert to Normal* turns your Easy settings into per-block values.
- **Anima Easy**: nine semantic sliders for Anima (Hands / Line quality, Body / Clothing, Pose / Composition, Face structure, Eyes / Expression, Mouth / Lower face, Background / Detail density, Style / Texture, Color / Final look), based on Team-C's per-layer breakdown. They also work on Krea 2 and MOSS-Image, but are calibrated on Anima.
- **Anima**: one slider per DiT block, for Anima, Krea 2 and MOSS-Image. The slider count follows Model A's real depth.

**Depth-upscaled Anima**: when Model A and Model B have a different number of blocks (for example 3.8B with the base 2B), NeoMerger lines up the matching blocks using the model's metadata or the known upscale recipes. You choose what happens to the blocks that exist only in A.

**Method tuning**: Filter strength, outlier cut, magnitude bias and row-wise NuSLERP, shown only for the methods that use them. **Legacy mode** reproduces merges made before per-block weights worked on task-vector methods.

<img width="686" height="841" alt="neomergeranalyzerui" src="https://github.com/user-attachments/assets/fd9398bc-d833-4eeb-b6c6-2d9ac1c04ddc" />

**Block Similarity Analyzer**: before merging, see how different the two models are per block, from blue (nearly identical) to red (very different). Works on SDXL and on DiT models.

**Merge & Gen**: merges, loads the result and generates a preview with the prompt and settings of the txt2img tab, shown inside NeoMerger.

**Alpha Sweep**: enter a list or range of alphas (for example `0.2-0.8:0.15`, up to 16 steps). NeoMerger merges and generates one image per value on the same seed, then builds a comparison grid. Intermediate models are deleted afterwards unless you keep them.

---

### LoRA Tools tab

<img width="1834" height="947" alt="neomergerloraui" src="https://github.com/user-attachments/assets/7d81b6ea-d1fe-4758-8c9a-89c6ab995d22" />

**Bake-in**: permanently fuse a LoRA into a checkpoint. Easy mode uses one strength slider (0–2). Advanced mode gives per-category control on SDXL. On DiT models it falls back to a single strength.
- Formats: kohya (`lora_down` / `lora_up`), PEFT / Diffusers (`lora_A` / `lora_B`), LyCORIS LoKr, DoRA
- SDXL LoRAs saved with Diffusers block names are converted automatically
- For an Anima LoRA trained on a different depth, pick its base model in *LoRA reference model* so the layers land in the right place
- After every bake, the terminal prints a match report: how many LoRA modules were fused and why any were skipped

**LoRA Merge**: combine two LoRA files with a blend slider. Different ranks are handled automatically.

**Extract**: build a LoRA from *finetuned − base*. Choose the rank, an optional conv rank (LoCon, SDXL only) and whether to include the text encoders. SVD runs on the GPU when available. The result is a standard kohya LoRA, usable anywhere.

---

### Inspect tab

<img width="1828" height="925" alt="neomergerinspectui" src="https://github.com/user-attachments/assets/abea4721-32b6-48da-b001-f7f693ca5a7d" />

Load any model or LoRA to inspect:
- File size, SHA256 hash, detected architecture
- Total keys and parameters, precision breakdown
- Embedded metadata: NeoMerger recipe, SuperMerger recipe, LoRA training info, trigger words

Inspect reads only the file header, so it is instant even on huge files. You can also clear all metadata from a file.

---

### Block Probe tab

Swaps blocks from Model B into the model currently loaded in the WebUI and generates an image for each, using your txt2img prompt and settings. Three modes: one block per image, one image per semantic group, or all selected blocks together. Everything happens in memory: no file is written and Model A is restored after every image.

---

### Advanced Options and Settings

Available in every tab that writes a file:
- **Output precision**: fp16 (default), bf16, fp32, fp8. Use **bf16** for Anima, Krea 2 and MOSS-Image: NeoMerger warns you when a bf16 model is about to be saved as fp16
- **VAE swap**: replace the VAE before saving
- **Save metadata**: embed the recipe in the output file
- **Norm clamp**: pulls the merged weights back toward Model A's magnitude, against the washed-out or burnt look of chained merges

In **Settings → NeoMerger** you choose when the output name is generated automatically, and what happens when the file already exists. Input models are never overwritten.

---

## System Requirements

- **RAM**: `.safetensors` models are read piece by piece and the output is streamed to disk, so RAM no longer depends on the model size (`.ckpt` files are still loaded whole). 16 GB is comfortable for every supported architecture. Before each merge the terminal shows a RAM estimate
- **Disk**: as much free space as the output file: about 7 GB for SDXL in fp16, about 26 GB for Krea 2 in bf16. NeoMerger checks this before starting
- The merge writes to a `.part` file that is renamed only at the end, so a stopped or failed merge never leaves a broken model behind

---

## Presets

Presets are saved as JSON files in `extensions/neomerger/presets/`. You can back them up, share them, or commit them to your own fork.

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

- [Vetehine](https://civitai.com/user/Vetehine): testing, feedback and bug reports
- [EarthboundAI](https://civitai.com/user/EarthboundAI): testing, feedback and bug reports

---

## Author

**AkiumAI** · [github.com/AkiumAI](https://github.com/AkiumAI) · [CivitAI](https://civitai.com/user/Akium)

---

## Support the project

If NeoMerger has been useful to you, consider supporting continued development on Ko-fi.
Every contribution helps keep this project free, open-source, and non-commercial.

<a href="https://ko-fi.com/akinak4">
  <img src="https://github.com/user-attachments/assets/24adecb9-27d9-451b-ba92-ad3d0f607d7c" alt="Support NeoMerger on Ko-fi" width="100%" />
</a>

Thank you! ☕

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — see [LICENSE](LICENSE)

You are free to use, share, and modify this project for **non-commercial purposes** with attribution.
Commercial use is not permitted without explicit written permission from the author.

> This extension is provided as-is, without warranty of any kind. The author is not responsible
> for any damages or data loss resulting from its use. Always keep backups of your original models
> before merging.
