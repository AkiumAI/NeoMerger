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
---
## [1.2.0]
**Added**
- Anima block merge — 28 DiT blocks (BLK_00 to BLK_27) in a dedicated mode
- Anima LoRA bake-in — fuses UNet LoRA layers into Anima checkpoints
- Automatic architecture detection at merge time (SDXL vs Anima)
- Bake-in now logs detected architecture to terminal
**Fixed**
- Anima checkpoints with embedded VAE were misdetected as SDXL — `detect_arch()` now scans all keys instead of only the first 20, since `first_stage_model.*` (VAE) keys can appear at the start and hide the discriminating `net.blocks.*` keys further down. The fix applies everywhere `detect_arch()` is called: Block Merger, Bake-in, Block Similarity Analyzer, and the Inspect metadata reader.
**Notes**
- Anima TE keys (Qwen3 text encoder) are skipped during bake-in — the text encoder is a separate file not included in the checkpoint
- Anima LoRA bake-in: layer count varies by training config (UNet-only ~280 layers, UNet+TE ~476, partial-block training even fewer) — all are valid
- Block labels for Anima are positional only (BLK_00–BLK_27) — no documentation exists yet on what each block controls visually
---
## [1.3.0]
**Added**
- **Merge & Gen button** on the Merge tab — runs the merge, loads the merged model into the WebUI, and generates a preview image using the prompt and settings from the txt2img tab. The result is shown in an accordion inside NeoMerger without needing to switch tabs. Generation runs without txt2img extensions (ADetailer, HiRes Fix, etc.) — it's a clean preview of the merged base model; use the txt2img tab for refined generation.
- **Convert to Normal button** in Easy mode — expands the semantic sliders into the 20 per-block weights and switches to Normal mode for fine-tuning. No loss of fidelity — uses the same calculation the merge engine applies internally.
- **"Include experimental sliders" checkbox** in the Experimental accordion — when off (default), experimental sliders are skipped entirely from the weighted-average calculation so they don't dilute the main category values. State is saved with presets.
- **Base Alpha slider** in the no-blocks panel — controls the base block (text encoder · CLIP) independently from the global Alpha which is applied to the UNet. Especially useful for TIES and DARE methods, which don't support per-block weights. The value is also saved with presets and embedded in checkpoint metadata.
- JavaScript helper at `javascript/neomerger.js` to read txt2img fields for Merge & Gen.
**Changed**
- Default behavior of Easy mode no longer dilutes block weights with experimental categories. Existing presets keep their old behavior on load (toggle defaults to off, matching how those presets were used before).
**Notes**
- Old presets (pre-1.3.0) load fine — `use_exp` defaults to off and `base_alpha` defaults to 0.5.
---
[1.3.1]
Fixed

- LoRA Bake-in crashed on LoRAs containing convolutional layers with non-1×1 kernels (RuntimeError: Expected size for first two dimensions of batch2 tensor to be [...]). The matmul reconstruction now handles Conv2d layers with both the standard kohya convention (down: [rank, in_ch, kh, kw]) and the inverted-rank variant some trainers produce.
- Individual layers whose shape can't be reconciled are now skipped with a ⚠️ Skipped <layer> warning instead of aborting the whole bake — the rest of the LoRA still fuses.
