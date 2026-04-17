# NeoMerger — Changelog

**AkiumAI** — https://github.com/AkiumAI

---

## [1.0.0]

**Merge tab**
- Methods: Weighted Sum, SLERP, Add Difference, TIES, DARE
- Easy mode with 6 semantic sliders — Style, Anatomy, Face Details, Background, NSFW, Detail/Sharpness
- Experimental sliders in collapsed section — Saturation, Contrast, Brightness, Sharpness, Lights & Darkness
- Normal mode — all 20 SDXL blocks individually (BASE, IN00–IN08, MID, OUT00–OUT08)
- Per-block descriptions toggle, sourced from Crody's Model Merge Guide
- Block merge can be enabled/disabled independently from the merge method
- Add Difference shows a third model field automatically
- TIES and DARE lock block merge off automatically
- Block Similarity Analyzer — per-block difference chart with color coding and semantic tags
- Advanced Options: output precision, VAE swap, metadata toggle
- Preset system — save, load, delete

**LoRA Tools tab**
- Bake-in with Easy (global strength) and Advanced (per-category) modes
- LoRA Merge with automatic rank padding
- Advanced Options and presets on both sub-tabs

**Inspect tab**
- Architecture detection: SDXL, Pony, Illustrious, SD 1.x/2.x, Flux, SD3, LoRA types, LyCORIS, VAE, Textual Inversion
- Shows file size, SHA256, key count, parameter count, precision breakdown
- Reads embedded metadata: NeoMerger recipe, Supermerger, modelspec, kohya_ss training info, trigger words
- Clear metadata with atomic temp-file write

**General**
- Works on Forge Neo, Forge, reForge, A1111
- CPU-only — no VRAM needed
- Metadata embedded in every output
- Crash-safe saves

---

## [1.1.0]

**Added**
- Terminal logging with `[NeoMerger]` prefix on all operations
- tqdm progress bar during checkpoint merges
- Stop button on Merge, Bake-in, and LoRA Merge tabs

**Fixed**
- Preset dropdown was showing internal prefix (`bm_`) in names
- Preset load crashed with `ValueError` on old preset format
- Block Similarity Analyzer was showing ~99% similarity on same-family models — now uses relative difference instead of cosine similarity
- `TypeError` on Forge base / A1111 (Gradio 3.x) when slider values come in as strings
