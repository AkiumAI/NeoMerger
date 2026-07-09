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

[1.3.2]
**Added**

- PEFT / Diffusers LoRA format support (`lora_A.weight` / `lora_B.weight` naming) in Bake-in, alongside the existing kohya/A1111 convention (`lora_down` / `lora_up`). Covers diffusers-trained and ai-toolkit LoRAs, including native Anima DiT LoRAs (`diffusion_model.` prefix). PEFT LoRAs store no `.alpha`, so scale defaults to 1.0.
- Diffusers → kohya key-name conversion for SDXL LoRAs — trainers that save Diffusers-style block names (`lora_unet_down_blocks_*`) are remapped to kohya naming so they bake in correctly.

---
## [1.4.0]
**Added**
- Four new merge methods: **Task Arithmetic** (plain task-vector sum, supports negative alpha to *subtract* a style), **Breadcrumbs** (drops both outliers and noise, best for chained merges), **DELLA** (DARE with magnitude-ranked drop probabilities), **NuSLERP** (normalised SLERP with optional row-wise mode, supports block merge)
- Method tuning controls — Filter strength slider plus an advanced accordion (DELLA epsilon, Breadcrumbs gamma, NuSLERP row-wise). Alpha slider extends to −1…+1 on task-vector methods
- ⚡ Quick actions — "Add style" / "Remove style" one-click Breadcrumbs setup
- 🧪 **Block Probe tab** — swaps blocks in memory (no disk merge) and generates one preview image per block / semantic group / selection, using the current txt2img settings, with alpha blend and auto arch detection. Requires the updated `javascript/neomerger.js`
- Cross-prefix key matching (merge engine + Block Similarity) — Anima models saved under different prefixes (`net.` / `diffusion_model.` / `model.diffusion_model.`) now merge correctly; previously the merge silently kept only Model A. Key-match report in the terminal, with a warning below 50%
- Block Similarity is now architecture-aware — Anima models use their 28 DiT blocks instead of collapsing into one
- Inspect: "Anima (DiT)" architecture label

**Fixed**
- Anima LoRA bake ignored the Strength slider on blocks 20–27 (weight list was hard-coded to 20 entries) — weights are now built per-architecture. Advanced mode on Anima falls back to flat Strength with a warning, since the semantic categories are SDXL-only
- kohya-naming Anima LoRAs (`lora_unet_blocks_N_...`): compound module names were broken by a blanket `_ → .` replace (`cross_attn` → `cross.attn`) so those layers were silently skipped — tails are now rebuilt around known compound tokens and mapped to their real block index
- `get_anima_block_index` only recognised `net.blocks.` — other prefixes made every key fall to block 0 in Block Similarity
- Bake-in forced fp16 on fused layers regardless of the selected output precision — now respects fp16/bf16/fp32/fp8 (bit-identical for fp16)
- Integer tensors (e.g. int64 `position_ids`) were corrupted by the precision cast on save — non-floating tensors are now preserved
- Corrupted byte in the `ff_net` regex replacement (literal `0x01` instead of `\1`) — dead code path, fixed for safety

**Performance**
- Lazy per-key tensor reading in the merge engine, Block Similarity and Block Probe — no more loading whole checkpoints into RAM (analyzing two SDXL models previously needed ~13 GB)
- TIES / DARE / Breadcrumbs thresholds via `torch.kthvalue` instead of `torch.quantile` — exact at every size (quantile *sampled* above 16M elements) and ~8× faster: typical TIES merge ~800 s → ~200 s
- DELLA ranks large tensors via quantile buckets instead of a full argsort; NuSLERP row-wise fully vectorised
- Inspect is instant on any file size (reads the safetensors header, no tensor loading) and the SHA256 hash is cached until the file changes
- Saving no longer builds a second cast copy of the state dict (lower peak RAM, mainly for fp32 sources)

**Removed**
- Dead `merge_checkpoints()` function, superseded by the unified merge engine

**Notes**
- DARE and DELLA drop weights at random by design, so running the same merge twice gives slightly different (statistically equivalent) results — this has always been true for DARE
- Block merge remains unavailable for TIES, DARE, Task Arithmetic, Breadcrumbs and DELLA; NuSLERP supports it
- Old presets load fine — no format changes
