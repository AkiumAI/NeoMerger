# Changelog

All notable changes to NeoMerger are documented here.

Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

---

## [1.0.0] — Initial Release

### Added
- **Block Merge** tab — Easy mode (semantic sliders) and Normal mode (per-block control)
- **LoRA Bake-in** — fuse a LoRA into a checkpoint with Easy (global strength) or Advanced (per-category) mode
- **LoRA Merge** — combine two LoRA files with automatic rank padding
- **Model Inspector** tab — architecture detection, metadata reader, merge recipe viewer
- **Advanced Options** in all merge tabs — precision (fp16/bf16/fp32/fp8), VAE swap, metadata toggle
- **Presets** — save and load merge configurations as JSON
- **Metadata embedding** — NeoMerger writes merge recipes inside output files
- Experimental semantic sliders: Saturation, Contrast, Brightness, Sharpness, Lights & Darkness
- Architecture detection for SDXL, Pony, Illustrious, SD 1.x, SD 2.x, Flux, SD 3.x, LoRA, LyCORIS, VAE, Textual Inversion
- Metadata reader for NeoMerger, Supermerger, modelspec, kohya_ss LoRA training info, trigger words
- Compatible with Forge Neo, Forge, reForge, AUTOMATIC1111
