"""
NeoMerger — Advanced model merging extension
Developed by AkiumAI — https://github.com/AkiumAI

Supports: Forge Neo, Forge, reForge, AUTOMATIC1111
Tabs: Merge · LoRA Tools · Inspect

License: CC BY-NC 4.0
https://creativecommons.org/licenses/by-nc/4.0/
© 2025 AkiumAI
"""

import os
import json
import torch
import gradio as gr
from pathlib import Path

from modules import script_callbacks, shared
from modules.paths import models_path, data_path

# Global stop flag — set to True to interrupt any running merge
_STOP_MERGE = [False]

def _log(msg: str):
    """Print a prefixed message to the Forge terminal."""
    print(f"[NeoMerger] {msg}")

# Ensure Forge tmp directory exists for Gradio gallery
try:
    _forge_tmp = os.path.join(data_path, "tmp")
    os.makedirs(_forge_tmp, exist_ok=True)
except Exception:
    # Fallback to common install paths
    for _p in [
        "/workspace/stable-diffusion-webui-forge/tmp",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tmp"),
    ]:
        try:
            os.makedirs(_p, exist_ok=True)
            break
        except Exception:
            pass

# ═════════════════════════════════════════════════════════════════════════════
#  PATH UTILS
# ═════════════════════════════════════════════════════════════════════════════

def get_ckpt_dir() -> str:
    if hasattr(shared, "cmd_opts"):
        opts = shared.cmd_opts
        if hasattr(opts, "ckpt_dirs") and opts.ckpt_dirs:
            return opts.ckpt_dirs[0]
        if hasattr(opts, "ckpt_dir") and opts.ckpt_dir:
            return opts.ckpt_dir
    return os.path.join(models_path, "Stable-diffusion")

def get_lora_dir() -> str:
    if hasattr(shared, "cmd_opts"):
        opts = shared.cmd_opts
        if hasattr(opts, "lora_dir") and opts.lora_dir:
            return opts.lora_dir
    return os.path.join(models_path, "Lora")

def get_model_list():
    d = get_ckpt_dir()
    out = []
    for ext in ["*.safetensors", "*.ckpt"]:
        out += [os.path.basename(str(p)) for p in Path(d).rglob(ext)]
    return sorted(out)

def get_lora_list():
    d = get_lora_dir()
    out = []
    for ext in ["*.safetensors", "*.pt", "*.ckpt"]:
        out += [os.path.basename(str(p)) for p in Path(d).rglob(ext)]
    return sorted(out)

def find_in_dir(name: str, directory: str) -> str:
    for root, _, files in os.walk(directory):
        for f in files:
            if f == name:
                return os.path.join(root, f)
    return name

def find_model(name): return find_in_dir(name, get_ckpt_dir())
def find_lora(name):  return find_in_dir(name, get_lora_dir())

# ═════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE MAPS
# ═════════════════════════════════════════════════════════════════════════════

# ── Checkpoint blocks ────────────────────────────────────────────────────────
CKPT_BLOCK_KEYS = (
    ["base"] +
    [f"input_block_{i}" for i in range(9)] +
    ["middle_block"] +
    [f"output_block_{i}" for i in range(9)]
)
CKPT_BLOCK_LABELS = (
    ["BASE"] +
    [f"IN{i:02d}" for i in range(9)] +
    ["MID"] +
    [f"OUT{i:02d}" for i in range(9)]
)

def get_ckpt_block_index(key: str) -> int:
    if "model.diffusion_model" not in key:
        return 0
    if "middle_block" in key:
        return 10
    if "input_blocks" in key:
        try: return 1 + int(key.split("input_blocks.")[1].split(".")[0])
        except: return 0
    if "output_blocks" in key:
        try: return 11 + int(key.split("output_blocks.")[1].split(".")[0])
        except: return 0
    return 0

# ── LoRA blocks (SDXL naming) ────────────────────────────────────────────────
# SDXL UNet LoRA key patterns → logical block index (same 20-slot schema)
LORA_BLOCK_PATTERNS = [
    # (pattern_substring, block_index)
    # BASE — text encoder + misc
    ("lora_te",                          0),
    # Input blocks
    ("lora_unet_down_blocks_0_attentions_0", 1),
    ("lora_unet_down_blocks_0_attentions_1", 2),
    ("lora_unet_down_blocks_1_attentions_0", 3),
    ("lora_unet_down_blocks_1_attentions_1", 4),
    ("lora_unet_down_blocks_2_attentions_0", 5),
    ("lora_unet_down_blocks_2_attentions_1", 6),
    # Extra down resnets map to nearest attention block
    ("lora_unet_down_blocks_0",          1),
    ("lora_unet_down_blocks_1",          3),
    ("lora_unet_down_blocks_2",          5),
    # Middle
    ("lora_unet_mid_block",              10),
    # Output blocks
    ("lora_unet_up_blocks_0_attentions_0", 11),
    ("lora_unet_up_blocks_0_attentions_1", 12),
    ("lora_unet_up_blocks_0_attentions_2", 13),
    ("lora_unet_up_blocks_1_attentions_0", 14),
    ("lora_unet_up_blocks_1_attentions_1", 15),
    ("lora_unet_up_blocks_1_attentions_2", 16),
    ("lora_unet_up_blocks_2_attentions_0", 17),
    ("lora_unet_up_blocks_2_attentions_1", 18),
    ("lora_unet_up_blocks_2_attentions_2", 19),
    ("lora_unet_up_blocks_0",            11),
    ("lora_unet_up_blocks_1",            14),
    ("lora_unet_up_blocks_2",            17),
]

def get_lora_block_index(key: str) -> int:
    key_lower = key.lower()
    for pattern, idx in LORA_BLOCK_PATTERNS:
        if pattern in key_lower:
            return idx
    return 0  # fallback: BASE

# ═════════════════════════════════════════════════════════════════════════════
#  SEMANTIC CATEGORIES  (shared between Block Merge and LoRA Tools)
# ═════════════════════════════════════════════════════════════════════════════

EASY_CATEGORIES = {
    "Style / Colors": {
        "icon": "🎨",
        "description": "Artistic look, color palette, brushwork",
        "blocks": {
            "base": 0.8,
            "input_block_0": 0.7, "input_block_1": 0.6,
            "output_block_7": 0.6, "output_block_8": 0.8,
        }
    },
    "Anatomy / Proportions": {
        "icon": "🦴",
        "description": "Body proportions, posture, structural correctness",
        "blocks": {
            "input_block_4": 0.9, "input_block_5": 0.9,
            "middle_block": 0.7,
            "output_block_3": 0.8, "output_block_4": 0.9,
        }
    },
    "Face Details": {
        "icon": "👁️",
        "description": "Face quality, eyes, expressions",
        "blocks": {
            "input_block_6": 0.9, "input_block_7": 0.8,
            "output_block_1": 0.9, "output_block_2": 0.8,
        }
    },
    "Background / Scene": {
        "icon": "🌄",
        "description": "Background, environment, perspective, composition",
        "blocks": {
            "input_block_2": 0.8, "input_block_3": 0.7,
            "output_block_5": 0.7, "output_block_6": 0.8,
        }
    },
    "NSFW / Mature Content": {
        "icon": "🔞",
        "description": "Transfers mature content characteristics from Model B",
        "blocks": {
            "input_block_5": 0.95, "input_block_6": 0.9,
            "middle_block": 0.85,
            "output_block_2": 0.9, "output_block_3": 0.95,
        }
    },
    "Detail / Sharpness": {
        "icon": "✨",
        "description": "Fine detail, texture, overall sharpness",
        "blocks": {
            "input_block_8": 0.8,
            "output_block_0": 0.9, "output_block_1": 0.85,
        }
    },

    # ── Experimental sliders ─────────────────────────────────────────────────
    # These do NOT map cleanly to specific blocks — results vary between models.
    # Mappings are heuristic approximations based on community observations.

    "⚗️ [EXP] Saturation": {
        "icon": "🎨",
        "description": "[Experimental] Color richness and vibrancy — unreliable, varies by model",
        "blocks": {
            "base": 0.7,
            "input_block_0": 0.8, "input_block_1": 0.7,
            "output_block_7": 0.7, "output_block_8": 0.9,
        }
    },
    "⚗️ [EXP] Contrast": {
        "icon": "◑",
        "description": "[Experimental] Tonal range between lights and darks — unreliable, varies by model",
        "blocks": {
            "input_block_2": 0.7, "input_block_3": 0.6,
            "middle_block": 0.6,
            "output_block_5": 0.6, "output_block_6": 0.7,
        }
    },
    "⚗️ [EXP] Brightness": {
        "icon": "☀️",
        "description": "[Experimental] Overall image brightness — unreliable, varies by model",
        "blocks": {
            "base": 0.5,
            "input_block_0": 0.6,
            "output_block_7": 0.6, "output_block_8": 0.7,
        }
    },
    "⚗️ [EXP] Sharpness": {
        "icon": "🔪",
        "description": "[Experimental] Edge crispness and definition — unreliable, varies by model",
        "blocks": {
            "input_block_7": 0.7, "input_block_8": 0.8,
            "output_block_0": 0.9, "output_block_1": 0.8,
        }
    },
    "⚗️ [EXP] Lights & Darkness": {
        "icon": "🌗",
        "description": "[Experimental] Shadow depth and highlight intensity — unreliable, varies by model",
        "blocks": {
            "input_block_3": 0.6, "input_block_4": 0.6,
            "middle_block": 0.7,
            "output_block_4": 0.6, "output_block_5": 0.7,
        }
    },
}

def easy_to_block_weights(slider_values: dict) -> list:
    """Semantic sliders → 20 per-block weights."""
    contributions = [0.0] * 20
    counts        = [0]   * 20
    for cat_name, val in slider_values.items():
        if cat_name not in EASY_CATEGORIES:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        for block_name, base_w in EASY_CATEGORIES[cat_name]["blocks"].items():
            if block_name not in CKPT_BLOCK_KEYS:
                continue
            idx = CKPT_BLOCK_KEYS.index(block_name)
            contributions[idx] += base_w * val
            counts[idx]        += 1
    return [
        min(1.0, contributions[i] / counts[i]) if counts[i] > 0 else 0.0
        for i in range(20)
    ]

# ═════════════════════════════════════════════════════════════════════════════
#  PRESETS
# ═════════════════════════════════════════════════════════════════════════════

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

def list_presets(prefix=""):
    # Strip the prefix from display names so dropdown shows clean names
    return [f.stem[len(prefix):] for f in Path(PRESETS_DIR).glob(f"{prefix}*.json")]

def save_preset(name, data, prefix=""):
    with open(os.path.join(PRESETS_DIR, f"{prefix}{name}.json"), "w") as f:
        json.dump(data, f, indent=2)

def load_preset(name, prefix=""):
    p = os.path.join(PRESETS_DIR, f"{prefix}{name}.json")
    return json.load(open(p)) if os.path.isfile(p) else {}

def delete_preset(name, prefix=""):
    p = os.path.join(PRESETS_DIR, f"{prefix}{name}.json")
    if os.path.isfile(p): os.remove(p)

# ═════════════════════════════════════════════════════════════════════════════
#  MERGE ENGINES
# ═════════════════════════════════════════════════════════════════════════════

def load_sd(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)

def cast_precision(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Cast tensor to requested precision."""
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8":  torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    }
    dtype = mapping.get(precision, torch.float16)
    return tensor.to(dtype=dtype)

def save_sf(state_dict: dict, path: str, precision: str = "fp16", metadata: dict = None):
    """Save state dict as safetensors with optional precision cast and metadata."""
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cast = {k: cast_precision(v, precision).contiguous() for k, v in state_dict.items()}
    meta = {k: str(v) for k, v in metadata.items()} if metadata else {}
    save_file(cast, path, metadata=meta)

def compute_hash(path: str) -> str:
    """Compute short SHA256 hash of a file (first 8 hex chars)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

def get_vae_list() -> list:
    """List available VAE files."""
    vae_dir = os.path.join(models_path, "VAE")
    vaes = []
    for ext in ["*.safetensors", "*.pt", "*.ckpt"]:
        vaes += [os.path.basename(str(p)) for p in Path(vae_dir).rglob(ext)]
    return sorted(vaes)

def find_vae(name: str) -> str:
    vae_dir = os.path.join(models_path, "VAE")
    for root, _, files in os.walk(vae_dir):
        for f in files:
            if f == name:
                return os.path.join(root, f)
    return name

def swap_vae(state_dict: dict, vae_path: str) -> dict:
    """Replace VAE weights in state_dict with those from vae_path."""
    vae_sd = load_sd(vae_path)
    VAE_PREFIX = "first_stage_model."
    result = {k: v for k, v in state_dict.items() if not k.startswith(VAE_PREFIX)}
    for k, v in vae_sd.items():
        key = k if k.startswith(VAE_PREFIX) else VAE_PREFIX + k
        result[key] = v
    del vae_sd
    return result

# ── 1. Checkpoint block merge ────────────────────────────────────────────────

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0f = v0.float().flatten()
    v1f = v1.float().flatten()
    dot  = (v0f * v1f).sum() / (v0f.norm() * v1f.norm() + eps)
    dot  = dot.clamp(-1, 1)
    # If nearly parallel, fall back to linear interpolation
    if dot.abs() > 0.9995:
        return ((1 - t) * v0 + t * v1)
    theta = dot.acos()
    s0    = (((1 - t) * theta)).sin()
    s1    = ((t * theta)).sin()
    st    = theta.sin()
    return ((s0 / st) * v0.float() + (s1 / st) * v1.float())

def _quantile_safe(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """
    Safe quantile computation that handles tensors larger than PyTorch's
    quantile() limit (~16M elements) by sampling.
    """
    flat = tensor.abs().flatten()
    n    = flat.numel()
    # PyTorch quantile limit is 2^24 = 16,777,216
    if n <= 16_000_000:
        return flat.quantile(q)
    # Sample ~10M elements for large tensors
    idx = torch.randperm(n, device=flat.device)[:10_000_000]
    return flat[idx].quantile(q)

def ties_merge(tensors: list, alphas: list, density: float = 0.5) -> torch.Tensor:
    """
    TIES merge: Trim, Elect Sign, Merge.
    tensors: list of (base, other) pairs
    alphas:  per-tensor weights
    density: fraction of weights to keep (trim threshold)
    """
    base = tensors[0][0].float()
    task_vectors = [(t.float() - base) * a for (_, t), a in zip(tensors, alphas)]

    trimmed = []
    for tv in task_vectors:
        threshold = _quantile_safe(tv, 1.0 - density)
        trimmed.append(tv * (tv.abs() >= threshold))

    sign_sum = sum(t.sign() for t in trimmed)
    elected  = sign_sum.sign()

    merged = torch.zeros_like(base)
    count  = torch.zeros_like(base)
    for t in trimmed:
        agree  = (t.sign() == elected) | (elected == 0)
        merged = merged + t * agree
        count  = count  + agree.float()

    count = count.clamp(min=1)
    return base + merged / count

def dare_merge(base: torch.Tensor, others: list, alphas: list,
               density: float = 0.5) -> torch.Tensor:
    """
    DARE merge: Drop And REscale.
    Randomly drops weights from task vectors then rescales.
    """
    base_f = base.float()
    result = base_f.clone()
    for other, alpha in zip(others, alphas):
        tv   = other.float() - base_f
        mask = (torch.rand_like(tv) < density).float()
        tv_rescaled = tv * mask / (density + 1e-8)
        result = result + alpha * tv_rescaled
    return result

def merge_models_method(path_a, path_b, path_c, block_weights_b, global_alpha,
                        method, output_path, precision="fp16", vae_path=None,
                        metadata=None, cb=None):
    """
    Unified merge engine supporting all methods.
    path_c is only used for Add Difference.
    block_weights_b: per-block weights (20 values). Used by Weighted Sum and SLERP.
    global_alpha: fallback alpha for methods that don't support per-block.
    """
    import time
    _STOP_MERGE[0] = False
    t_start = time.time()

    _log(f"Starting merge — Method: {method} · Precision: {precision}")
    _log(f"Model A: {os.path.basename(path_a)}")
    _log(f"Model B: {os.path.basename(path_b)}")
    if path_c:
        _log(f"Model C: {os.path.basename(path_c)}")

    if cb: cb(0.05, "Loading Model A...")
    _log("Loading Model A...")
    sd_a = load_sd(path_a)

    if cb: cb(0.25, "Loading Model B...")
    _log("Loading Model B...")
    sd_b = load_sd(path_b)

    sd_c = None
    if method == "Add Difference" and path_c:
        if cb: cb(0.35, "Loading Model C...")
        _log("Loading Model C...")
        sd_c = load_sd(path_c)

    if cb: cb(0.50, "Merging...")

    result = {}
    keys   = list(sd_a.keys())
    total  = len(keys)

    try:
        from tqdm import tqdm
        pbar = tqdm(keys, desc=f"[NeoMerger] {method}", unit="key",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except ImportError:
        pbar = keys

    for ki, key in enumerate(pbar):
        if ki % 100 == 0 and _STOP_MERGE[0]:
            _log("⏹  Merge stopped by user.")
            del sd_a, sd_b
            if sd_c: del sd_c
            torch.cuda.empty_cache()
            if cb: cb(0.0, "⏹  Stopped by user.")
            return False

        t_a = sd_a[key].float()
        t_b = sd_b[key].float() if key in sd_b else t_a

        if method == "Weighted Sum":
            idx = get_ckpt_block_index(key)
            w_b = block_weights_b[idx] if idx < len(block_weights_b) else global_alpha
            result[key] = (1 - w_b) * t_a + w_b * t_b

        elif method == "SLERP":
            idx = get_ckpt_block_index(key)
            w_b = block_weights_b[idx] if idx < len(block_weights_b) else global_alpha
            if t_a.shape == t_b.shape and t_a.numel() > 1:
                result[key] = slerp(w_b, t_a, t_b).reshape(t_a.shape)
            else:
                result[key] = (1 - w_b) * t_a + w_b * t_b

        elif method == "Add Difference":
            t_c = sd_c[key].float() if (sd_c and key in sd_c) else t_a
            idx = get_ckpt_block_index(key)
            w   = block_weights_b[idx] if idx < len(block_weights_b) else global_alpha
            result[key] = t_a + w * (t_b - t_c)

        elif method == "TIES":
            result[key] = ties_merge(
                [(t_a, t_b)], [global_alpha]
            ).reshape(t_a.shape)

        elif method == "DARE":
            result[key] = dare_merge(t_a, [t_b], [global_alpha]).reshape(t_a.shape)

        else:
            result[key] = (1 - global_alpha) * t_a + global_alpha * t_b

        if cb and ki % 200 == 0:
            cb(0.5 + 0.40 * (ki / total), f"Keys: {ki}/{total}")

    del sd_a, sd_b
    if sd_c: del sd_c

    if vae_path:
        if cb: cb(0.90, "Swapping VAE...")
        _log("Swapping VAE...")
        result = swap_vae(result, vae_path)

    if cb: cb(0.95, f"Saving ({precision})...")
    _log(f"Saving → {os.path.basename(output_path)} ({precision})...")
    save_sf(result, output_path, precision=precision, metadata=metadata)
    del result
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    _log(f"✅ Done in {elapsed:.1f}s → {os.path.basename(output_path)}")
    if cb: cb(1.0, "Done!")
    return True

def merge_checkpoints(path_a, path_b, block_weights_b, output_path,
                      precision="fp16", vae_path=None, save_metadata=True, cb=None):
    if cb: cb(0.05, "Loading Model A...")
    sd_a = load_sd(path_a)
    if cb: cb(0.25, "Loading Model B...")
    sd_b = load_sd(path_b)
    if cb: cb(0.50, "Merging...")

    result = {}
    keys   = list(sd_a.keys())
    for ki, key in enumerate(keys):
        idx = get_ckpt_block_index(key)
        w_b = block_weights_b[idx] if idx < len(block_weights_b) else 0.5
        t_a = sd_a[key].float()
        t_b = sd_b[key].float() if key in sd_b else t_a
        result[key] = ((1 - w_b) * t_a + w_b * t_b)
        if cb and ki % 500 == 0:
            cb(0.5 + 0.40 * (ki / len(keys)), f"Keys: {ki}/{len(keys)}")

    del sd_a, sd_b

    if vae_path:
        if cb: cb(0.90, "Swapping VAE...")
        result = swap_vae(result, vae_path)

    meta = None
    if save_metadata:
        meta = {
            "neomerger_version":  "1.0",
            "merge_type":         "block_merge",
            "model_a":            os.path.basename(path_a),
            "model_b":            os.path.basename(path_b),
            "block_weights":      json.dumps([round(w, 4) for w in block_weights_b]),
            "precision":          precision,
            "vae":                os.path.basename(vae_path) if vae_path else "original",
        }

    if cb: cb(0.95, f"Saving ({precision})...")
    save_sf(result, output_path, precision=precision, metadata=meta)
    del result
    torch.cuda.empty_cache()
    if cb: cb(1.0, "Done!")

# ── 2. LoRA bake-in ──────────────────────────────────────────────────────────

def bake_lora_into_checkpoint(ckpt_path, lora_path, block_weights, output_path,
                              precision="fp16", vae_path=None, save_metadata=True, cb=None):
    """
    Fuses a LoRA into a checkpoint using per-block alpha weights.
    LoRA keys follow the pattern: <name>.lora_up.weight / lora_down.weight / alpha
    The effective delta is: (lora_up @ lora_down) * (alpha / rank) * block_weight
    """
    import time
    _STOP_MERGE[0] = False
    t_start = time.time()
    _log(f"Bake-in: {os.path.basename(lora_path)} → {os.path.basename(ckpt_path)}")

    if cb: cb(0.05, "Loading checkpoint...")
    _log("Loading checkpoint...")
    sd = load_sd(ckpt_path)
    if cb: cb(0.30, "Loading LoRA...")
    _log("Loading LoRA...")
    lora = load_sd(lora_path)

    if cb: cb(0.50, "Fusing LoRA weights...")
    _log("Fusing LoRA weights...")

    lora_map = {}
    for key in lora.keys():
        if "lora_down" in key:
            base = key.replace(".lora_down.weight", "").replace("lora_down.weight", "")
            up_key    = key.replace("lora_down", "lora_up")
            alpha_key = base + ".alpha"
            if up_key in lora:
                rank  = lora[key].shape[0]
                alpha = lora[alpha_key].item() if alpha_key in lora else float(rank)
                lora_map[base] = {
                    "down":  lora[key].float(),
                    "up":    lora[up_key].float(),
                    "scale": alpha / rank,
                }

    def lora_key_to_ckpt(lora_base: str) -> str:
        k = lora_base
        k = k.replace("lora_unet_", "model.diffusion_model.")
        k = k.replace("_down_blocks_", ".input_blocks.")
        k = k.replace("_up_blocks_",   ".output_blocks.")
        k = k.replace("_mid_block",    ".middle_block")
        k = k.replace("_attentions_",  ".")
        k = k.replace("_resnets_",     ".")
        k = k.replace("_", ".")
        return k + ".weight"

    fused = 0
    for base_key, parts in lora_map.items():
        if _STOP_MERGE[0]:
            _log("⏹  Bake-in stopped by user.")
            del sd, lora
            torch.cuda.empty_cache()
            if cb: cb(0.0, "⏹  Stopped by user.")
            return 0
        ckpt_key = lora_key_to_ckpt(base_key)
        if ckpt_key not in sd:
            continue

        block_idx = get_lora_block_index(base_key)
        w = block_weights[block_idx] if block_idx < len(block_weights) else 1.0
        if w < 1e-6:
            continue

        down  = parts["down"]
        up    = parts["up"]
        scale = parts["scale"] * w

        if down.dim() == 4:
            delta = (up.squeeze() @ down.squeeze()).reshape(sd[ckpt_key].shape)
        else:
            delta = up @ down

        sd[ckpt_key] = (sd[ckpt_key].float() + scale * delta).half()
        fused += 1

    if vae_path:
        if cb: cb(0.88, "Swapping VAE...")
        sd = swap_vae(sd, vae_path)

    meta = None
    if save_metadata:
        meta = {
            "neomerger_version": "1.0",
            "merge_type":        "lora_bake",
            "checkpoint":        os.path.basename(ckpt_path),
            "lora":              os.path.basename(lora_path),
            "block_weights":     json.dumps([round(w, 4) for w in block_weights]),
            "precision":         precision,
            "vae":               os.path.basename(vae_path) if vae_path else "original",
            "fused_layers":      str(fused),
        }

    elapsed = time.time() - t_start
    _log(f"Fused {fused} layers. Saving ({precision})...")
    if cb: cb(0.90, f"Fused {fused} LoRA layers. Saving ({precision})...")
    save_sf(sd, output_path, precision=precision, metadata=meta)
    del sd, lora
    torch.cuda.empty_cache()
    _log(f"✅ Done in {elapsed:.1f}s — {fused} layers fused → {os.path.basename(output_path)}")
    if cb: cb(1.0, "Done!")
    return fused

# ── 3. LoRA + LoRA merge ─────────────────────────────────────────────────────

def merge_loras(path_a, path_b, weight_a, output_path,
                precision="fp16", save_metadata=True, cb=None):
    """
    Merges two LoRAs into one.
    weight_a: how much of LoRA A to keep (0.0 = only B, 1.0 = only A).
    Handles rank differences by zero-padding the smaller LoRA.
    """
    import time
    t_start = time.time()
    _log(f"LoRA Merge: {os.path.basename(path_a)} + {os.path.basename(path_b)}")
    if cb: cb(0.05, "Loading LoRA A...")
    _log("Loading LoRA A...")
    la = load_sd(path_a)
    if cb: cb(0.30, "Loading LoRA B...")
    _log("Loading LoRA B...")
    lb = load_sd(path_b)
    if cb: cb(0.50, "Merging LoRAs...")
    _log("Merging LoRAs...")

    weight_b = 1.0 - weight_a
    result   = {}
    all_keys = set(la.keys()) | set(lb.keys())

    for key in all_keys:
        if _STOP_MERGE[0]:
            _log("⏹  LoRA merge stopped by user.")
            del la, lb
            torch.cuda.empty_cache()
            if cb: cb(0.0, "⏹  Stopped by user.")
            return
        ta = la.get(key)
        tb = lb.get(key)

        if ta is None:
            result[key] = tb.float() * weight_b
        elif tb is None:
            result[key] = ta.float() * weight_a
        else:
            fa = ta.float()
            fb = tb.float()
            if fa.shape != fb.shape:
                max_shape = tuple(max(a, b) for a, b in zip(fa.shape, fb.shape))
                fa_p = torch.zeros(max_shape)
                fb_p = torch.zeros(max_shape)
                fa_p[tuple(slice(0, s) for s in fa.shape)] = fa
                fb_p[tuple(slice(0, s) for s in fb.shape)] = fb
                fa, fb = fa_p, fb_p
            result[key] = weight_a * fa + weight_b * fb

    meta = None
    if save_metadata:
        meta = {
            "neomerger_version": "1.0",
            "merge_type":        "lora_merge",
            "lora_a":            os.path.basename(path_a),
            "lora_b":            os.path.basename(path_b),
            "weight_a":          str(round(weight_a, 4)),
            "weight_b":          str(round(weight_b, 4)),
            "precision":         precision,
        }

    _log(f"Saving merged LoRA ({precision})...")
    if cb: cb(0.90, f"Saving merged LoRA ({precision})...")
    save_sf(result, output_path, precision=precision, metadata=meta)
    del la, lb, result
    torch.cuda.empty_cache()
    elapsed = time.time() - t_start
    _log(f"✅ Done in {elapsed:.1f}s → {os.path.basename(output_path)}")
    if cb: cb(1.0, "Done!")

# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPERS  (shared slider blocks)
# ═════════════════════════════════════════════════════════════════════════════

def build_easy_sliders(prefix=""):
    """Returns dict of {cat_name: gr.Slider}"""
    sliders = {}

    # Standard sliders
    for cat_name, cat_info in EASY_CATEGORIES.items():
        if cat_name.startswith("⚗️ [EXP]"):
            continue
        with gr.Row():
            gr.Markdown(
                f"**{cat_info['icon']} {cat_name}**  "
                f"<small style='color:gray'>{cat_info['description']}</small>",
                scale=2
            )
            sliders[cat_name] = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05,
                label="", show_label=False, scale=3,
            )

    # Experimental sliders — collapsed by default
    with gr.Accordion("⚗️ Experimental sliders", open=False):
        gr.Markdown(
            "<small style='color:orange'>Results are unpredictable and vary between models. "
            "These do not map cleanly to specific model blocks.</small>"
        )
        gr.Markdown("")
        for cat_name, cat_info in EASY_CATEGORIES.items():
            if not cat_name.startswith("⚗️ [EXP]"):
                continue
            label = cat_name.replace("⚗️ [EXP] ", "")
            with gr.Row():
                gr.Markdown(
                    f"**{cat_info['icon']} {label}**  "
                    f"<small style='color:gray'>{cat_info['description'].split(' — ')[0]}</small>",
                    scale=2
                )
                sliders[cat_name] = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05,
                    label="", show_label=False, scale=3,
                )

    return sliders

# Block descriptions based on Crody's Model Merge Guide (civitai.com/articles/12245)
BLOCK_DESCRIPTIONS = {
    "base":           "Text encoder · handles prompt encoding and system type · ⚠️ set to 0 or 1 when mixing different base architectures (e.g. SDXL + Pony)",
    "input_block_0":  "IN00 · overall inputs, outlines, contrast and fingers · ⚠️ avoid float values to prevent finger errors",
    "input_block_1":  "IN01 · background, depth of field, hair and clothes details",
    "input_block_2":  "IN02 · anatomy details, hair and clothes",
    "input_block_3":  "IN03 · anatomy details, hair and clothes",
    "input_block_4":  "IN04 · textures, body anatomy and NSFW behavior",
    "input_block_5":  "IN05 · textures, body anatomy and NSFW behavior",
    "input_block_6":  "IN06 · clothing texture and details",
    "input_block_7":  "IN07 · pose structures",
    "input_block_8":  "IN08 · texture feel, depth and anatomy base",
    "middle_block":   "MID00 · overall similarity to Model B · also controls prompt adherence and drawing style",
    "output_block_0": "OUT00 · torso, face structure, age and eyes",
    "output_block_1": "OUT01 · upper body and face shape",
    "output_block_2": "OUT02 · body curves, nose and ears",
    "output_block_3": "OUT03 · skin color and mouth",
    "output_block_4": "OUT04 · overall texture and smooth lights",
    "output_block_5": "OUT05 · overall texture and smooth lights",
    "output_block_6": "OUT06 · overall texture and hard lights",
    "output_block_7": "OUT07 · overall colors and rough draw lines",
    "output_block_8": "OUT08 · output, colors and drawing lines",
}

def build_block_sliders(default=0.5):
    """Returns dict of {block_key: gr.Slider} — 20 sliders with optional descriptions."""
    sliders = {}

    with gr.Row():
        btn_a    = gr.Button("All → A (0.0)", size="sm", variant="secondary")
        btn_mid  = gr.Button("All → 0.5",     size="sm", variant="secondary")
        btn_b    = gr.Button("All → B (1.0)", size="sm", variant="secondary")
        btn_info = gr.Button("ℹ️ Show descriptions", size="sm", variant="secondary")

    desc_panels = {}

    def make_block_row(key, label, desc):
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                sliders[key] = gr.Slider(0.0, 1.0, value=default, step=0.05, label=label)
                with gr.Row(visible=False) as dp:
                    gr.Markdown(f"<small style='color:var(--color-text-secondary)'>{desc}</small>")
                desc_panels[key] = dp

    gr.Markdown("")
    gr.Markdown("**BASE**")
    make_block_row("base", "BASE", BLOCK_DESCRIPTIONS["base"])

    gr.Markdown("**Input Blocks**")
    for i in range(9):
        key = f"input_block_{i}"
        make_block_row(key, f"IN{i:02d}", BLOCK_DESCRIPTIONS[key])

    gr.Markdown("**Middle Block**")
    make_block_row("middle_block", "MID", BLOCK_DESCRIPTIONS["middle_block"])

    gr.Markdown("**Output Blocks**")
    for i in range(9):
        key = f"output_block_{i}"
        make_block_row(key, f"OUT{i:02d}", BLOCK_DESCRIPTIONS[key])

    sv  = list(sliders.values())
    dps = list(desc_panels.values())

    btn_a.click(fn=lambda: [gr.update(value=0.0)] * len(sv), outputs=sv)
    btn_b.click(fn=lambda: [gr.update(value=1.0)] * len(sv), outputs=sv)
    btn_mid.click(fn=lambda: [gr.update(value=0.5)] * len(sv), outputs=sv)

    show_state = [False]
    def toggle_descriptions():
        show_state[0] = not show_state[0]
        label = "ℹ️ Hide descriptions" if show_state[0] else "ℹ️ Show descriptions"
        return [gr.update(label=label)] + [gr.update(visible=show_state[0])] * len(dps)

    btn_info.click(fn=toggle_descriptions, outputs=[btn_info] + dps)

    return sliders

def build_preset_row(prefix=""):
    with gr.Group():
        gr.Markdown("### 📋 Presets")
        with gr.Row():
            name_in  = gr.Textbox(label="Name", placeholder="my_preset", scale=3)
            save_btn = gr.Button("Save", size="sm", scale=1, variant="secondary")
        with gr.Row():
            dd       = gr.Dropdown(list_presets(prefix), label="Saved presets", scale=3)
            load_btn = gr.Button("Load",  size="sm", scale=1)
            del_btn  = gr.Button("🗑",    size="sm", scale=1, variant="stop")
        status = gr.Markdown("")
    return name_in, save_btn, dd, load_btn, del_btn, status

# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:

        gr.Markdown("# 🔀 NeoMerger")
        gr.Markdown("Advanced model merging for **SDXL & Illustrious** · Forge Neo")
        gr.Markdown("<small style='color:var(--color-text-secondary)'>Developed by <strong>AkiumAI</strong> · <a href='https://github.com/AkiumAI' target='_blank'>github.com/AkiumAI</a> · CC BY-NC 4.0</small>")
        gr.Markdown("---")

        with gr.Tabs():

            # ═════════════════════════════════════════════════════════════
            #  TAB 1 — MERGE
            # ═════════════════════════════════════════════════════════════
            with gr.Tab("🔀  Merge"):

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=320):

                        # Models
                        with gr.Group():
                            gr.Markdown("### 📂 Models")
                            ml = get_model_list()
                            bm_model_a = gr.Dropdown(ml, label="Model A  (base)")
                            bm_model_b = gr.Dropdown(ml, label="Model B")
                            with gr.Row(visible=False) as bm_model_c_row:
                                bm_model_c = gr.Dropdown(ml, label="Model C  (reference for Add Difference)")
                            bm_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                            bm_refresh.click(
                                fn=lambda: [gr.update(choices=get_model_list())] * 3,
                                outputs=[bm_model_a, bm_model_b, bm_model_c]
                            )

                        gr.Markdown("")

                        # Difference map
                        with gr.Group():
                            gr.Markdown("### 📊 Block Similarity")
                            gr.Markdown(
                                "<small>Analyze how similar the two models are block by block. "
                                "Loads both models into RAM — same time as a merge.</small>"
                            )
                            bm_analyze_btn = gr.Button("🔍  Analyze similarity", variant="secondary")
                            bm_analyze_status = gr.Markdown("")
                            bm_diff_map = gr.HTML("")

                        gr.Markdown("")

                        # Merge method
                        with gr.Group():
                            gr.Markdown("### ⚗️ Merge Method")
                            bm_method = gr.Radio(
                                ["Weighted Sum", "SLERP", "Add Difference", "TIES", "DARE"],
                                value="Weighted Sum", label="",
                            )
                            bm_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                label="Alpha (global weight)",
                                info="Used by all methods. For block merge, overridden per-block.")
                            bm_method_info = gr.Markdown(
                                "<small>**Weighted Sum:** `(1-α)·A + α·B` — standard linear blend.</small>"
                            )

                        gr.Markdown("")

                        # Block merge toggle
                        with gr.Group():
                            gr.Markdown("### 🧱 Block Merge")
                            bm_use_blocks = gr.Checkbox(
                                value=False, label="Enable per-block weights",
                                info="Override alpha for individual blocks using Easy or Normal mode"
                            )
                            bm_blocks_disabled_msg = gr.Markdown(
                                "<small style='color:orange'>Block merge is not compatible with TIES and DARE — these methods work globally.</small>",
                                visible=False
                            )

                        gr.Markdown("")

                        # Output
                        with gr.Group():
                            gr.Markdown("### 💾 Output")
                            bm_out_name = gr.Textbox(label="File name", value="merged_model",
                                                     placeholder="e.g. nexus_abyss_mix")
                            gr.Markdown("<small>Saved as `.safetensors` in your checkpoints folder.</small>")

                        gr.Markdown("")

                        # Advanced Options
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            bm_precision = gr.Radio(
                                ["fp16", "bf16", "fp32", "fp8"], value="fp16",
                                label="Output precision",
                                info="fp16 = default · bf16 = better on some GPUs · fp32 = full quality (2x size) · fp8 = experimental"
                            )
                            with gr.Row():
                                bm_vae = gr.Dropdown(
                                    ["— keep original —"] + get_vae_list(),
                                    value="— keep original —",
                                    label="VAE swap", scale=4
                                )
                                bm_vae_ref = gr.Button("↻", size="sm", scale=1)
                                bm_vae_ref.click(
                                    fn=lambda: gr.update(choices=["— keep original —"] + get_vae_list()),
                                    outputs=[bm_vae]
                                )
                            bm_save_meta = gr.Checkbox(value=True, label="Save merge metadata")

                        gr.Markdown("")

                        # Presets
                        (bm_p_name, bm_p_save, bm_p_dd,
                         bm_p_load, bm_p_del, bm_p_status) = build_preset_row("bm_")

                        gr.Markdown("")
                        with gr.Row():
                            bm_run_btn  = gr.Button("🔀  Run Merge", variant="primary", scale=3)
                            bm_stop_btn = gr.Button("⏹  Stop",       variant="stop",    scale=1)
                        bm_status   = gr.Markdown("")
                        bm_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress", interactive=False)

                    with gr.Column(scale=2):

                        with gr.Group(visible=False) as bm_blocks_panel:
                            with gr.Group():
                                gr.Markdown("### ⚙️ Block Mode")
                                bm_mode = gr.Radio(
                                    ["Easy", "Normal"], value="Easy", label="",
                                    info="Easy = semantic categories  |  Normal = per-block control"
                                )

                            gr.Markdown("")

                            with gr.Group(visible=True) as bm_easy_panel:
                                gr.Markdown("### 🎛️ Semantic Sliders")
                                gr.Markdown("<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B</small>")
                                gr.Markdown("")
                                bm_easy_sliders = build_easy_sliders("bm_")

                            with gr.Group(visible=False) as bm_norm_panel:
                                gr.Markdown("### 🧱 Per-Block Weights")
                                gr.Markdown("<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B</small>")
                                gr.Markdown("")
                                bm_block_sliders = build_block_sliders(0.5)

                            bm_mode.change(
                                fn=lambda m: (gr.update(visible=m=="Easy"),
                                              gr.update(visible=m=="Normal")),
                                inputs=bm_mode,
                                outputs=[bm_easy_panel, bm_norm_panel]
                            )

                        with gr.Group(visible=True) as bm_no_blocks_panel:
                            gr.Markdown("### ℹ️ Block merge disabled")
                            gr.Markdown(
                                "Enable **per-block weights** on the left to control how each "
                                "part of the model blends. When disabled, the global Alpha slider is used."
                            )

                METHOD_INFO = {
                    "Weighted Sum":   "<small>**Weighted Sum:** `(1-α)·A + α·B` — standard linear blend. Most compatible.</small>",
                    "SLERP":          "<small>**SLERP:** Spherical interpolation — smoother transitions, fewer artifacts at extreme alpha values.</small>",
                    "Add Difference": "<small>**Add Difference:** `A + α·(B-C)` — transfers the difference between B and C onto A. Requires Model C.</small>",
                    "TIES":           "<small>**TIES:** Trims redundant weights before merging — produces cleaner results. Block merge not supported.</small>",
                    "DARE":           "<small>**DARE:** Drops and rescales weights randomly — reduces merge noise. Block merge not supported.</small>",
                }

                def on_method_change(method):
                    show_c        = method == "Add Difference"
                    block_ok      = method not in ("TIES", "DARE")
                    info          = METHOD_INFO.get(method, "")
                    return (
                        gr.update(visible=show_c),           # bm_model_c_row
                        gr.update(interactive=block_ok),     # bm_use_blocks
                        gr.update(visible=not block_ok),     # bm_blocks_disabled_msg
                        gr.update(value=info),               # bm_method_info
                        gr.update(visible=False),            # bm_blocks_panel (reset)
                        gr.update(visible=True),             # bm_no_blocks_panel (reset)
                        gr.update(value=False),              # bm_use_blocks reset
                    )

                bm_method.change(
                    fn=on_method_change,
                    inputs=[bm_method],
                    outputs=[bm_model_c_row, bm_use_blocks, bm_blocks_disabled_msg,
                             bm_method_info, bm_blocks_panel, bm_no_blocks_panel, bm_use_blocks]
                )

                def on_blocks_toggle(enabled, method):
                    if method in ("TIES", "DARE"):
                        return gr.update(visible=False), gr.update(visible=True)
                    return gr.update(visible=enabled), gr.update(visible=not enabled)

                bm_use_blocks.change(
                    fn=on_blocks_toggle,
                    inputs=[bm_use_blocks, bm_method],
                    outputs=[bm_blocks_panel, bm_no_blocks_panel]
                )

                bm_all = list(bm_easy_sliders.values()) + list(bm_block_sliders.values())

                def bm_run(method, alpha, use_blocks, mode, ma, mb, mc,
                           out_n, precision, vae, save_meta, *vals):
                    if not ma or not mb:
                        return "❌  Select both Model A and Model B.", 0.0
                    if method == "Add Difference" and not mc:
                        return "❌  Add Difference requires Model C.", 0.0

                    pa = find_model(ma)
                    pb = find_model(mb)
                    pc = find_model(mc) if mc else None
                    vp = find_vae(vae) if vae and vae != "— keep original —" else None
                    op = os.path.join(get_ckpt_dir(),
                                      (out_n.strip().replace(" ","_") or "merged_model") + ".safetensors")

                    ne = len(bm_easy_sliders)
                    if use_blocks and method not in ("TIES","DARE"):
                        if mode == "Easy":
                            bw = easy_to_block_weights(dict(zip(bm_easy_sliders.keys(), vals[:ne])))
                        else:
                            bw = [float(v) if v is not None else 0.5 for v in vals[ne:]]
                    else:
                        bw = [float(alpha)] * 20

                    meta = None
                    if save_meta:
                        meta = {
                            "neomerger_version": "1.0",
                            "merge_type":        "block_merge",
                            "merge_method":      method,
                            "model_a":           os.path.basename(pa),
                            "model_b":           os.path.basename(pb),
                            "model_c":           os.path.basename(pc) if pc else "N/A",
                            "alpha":             str(round(float(alpha), 4)),
                            "block_weights":     json.dumps([round(w, 4) for w in bw]),
                            "precision":         precision,
                            "vae":               os.path.basename(vp) if vp else "original",
                        }

                    try:
                        result = merge_models_method(pa, pb, pc, bw, float(alpha),
                                                     method, op, precision, vp, meta)
                        if result is False:
                            return "⏹  Merge stopped by user.", 0.0
                        return f"✅  Saved to `{op}`", 1.0
                    except Exception as e:
                        return f"❌  {e}", 0.0

                def bm_save_p(name, method, alpha, use_blocks, mode, *vals):
                    if not name.strip(): return "❌  Enter a name.", gr.update()
                    ne = len(bm_easy_sliders)
                    save_preset(name.strip(), {
                        "method": method, "alpha": alpha,
                        "use_blocks": use_blocks, "mode": mode,
                        "easy":   dict(zip(bm_easy_sliders.keys(),  vals[:ne])),
                        "normal": dict(zip(bm_block_sliders.keys(), vals[ne:])),
                    }, "bm_")
                    return f"✅  Saved **{name}**.", gr.update(choices=list_presets("bm_"))

                def bm_load_p(name):
                    if not name:
                        return ["❌  Select a preset first."] + [gr.update()] * (len(bm_all) + 4)
                    d = load_preset(name, "bm_")
                    if not d:
                        return [f"❌  Preset '{name}' not found. Try saving it again."] + [gr.update()] * (len(bm_all) + 4)

                    easy_data = d.get("easy", {})
                    ev = []
                    for k in bm_easy_sliders.keys():
                        v = easy_data.get(k, 0.0)
                        try:
                            ev.append(float(v))
                        except (TypeError, ValueError):
                            ev.append(0.0)

                    norm_data = d.get("normal", {})
                    nv = []
                    for k in bm_block_sliders.keys():
                        v = norm_data.get(k, 0.5)
                        try:
                            nv.append(float(v))
                        except (TypeError, ValueError):
                            nv.append(0.5)

                    _log(f"Loaded preset '{name}' — method: {d.get('method','Weighted Sum')}, "
                         f"easy keys: {len(easy_data)}, normal keys: {len(norm_data)}")

                    return ([f"✅  Loaded **{name}**.",
                             gr.update(value=d.get("method", "Weighted Sum")),
                             gr.update(value=float(d.get("alpha", 0.5))),
                             gr.update(value=bool(d.get("use_blocks", False))),
                             gr.update(value=d.get("mode", "Easy"))]
                            + [gr.update(value=v) for v in ev + nv])

                def bm_del_p(name):
                    delete_preset(name, "bm_")
                    return f"🗑️  Deleted **{name}**.", gr.update(choices=list_presets("bm_"))

                # Build reverse mapping: block_index → list of semantic labels
                BLOCK_SEMANTIC_LABELS = {}
                for cat_name, cat_info in EASY_CATEGORIES.items():
                    clean = cat_name.replace("⚗️ [EXP] ", "")
                    for block_name in cat_info["blocks"]:
                        if block_name in CKPT_BLOCK_KEYS:
                            idx = CKPT_BLOCK_KEYS.index(block_name)
                            BLOCK_SEMANTIC_LABELS.setdefault(idx, []).append(clean)

                def bm_analyze(ma, mb):
                    if not ma or not mb:
                        return "❌  Select both Model A and Model B.", ""
                    if ma == mb:
                        return "❌  Select two different models.", ""

                    try:
                        pa = find_model(ma)
                        pb = find_model(mb)
                        sd_a = load_sd(pa)
                        sd_b = load_sd(pb)
                    except Exception as e:
                        return f"❌  Could not load models: {e}", ""

                    # Compute relative difference per block
                    # diff = mean(|A-B| / (|A|+|B|+eps) * 2)
                    # 0% = identical, 100% = completely different
                    block_diffs  = {}
                    block_counts = {}

                    skipped_shape = 0
                    for key in sd_a.keys():
                        if key not in sd_b:
                            continue
                        t_a = sd_a[key]
                        t_b = sd_b[key]
                        # Skip if shapes differ — incompatible architectures
                        if t_a.shape != t_b.shape:
                            skipped_shape += 1
                            continue
                        idx   = get_ckpt_block_index(key)
                        t_af  = t_a.float()
                        t_bf  = t_b.float()
                        diff  = (t_af - t_bf).abs()
                        denom = (t_af.abs() + t_bf.abs() + 1e-8)
                        rel   = (diff / denom * 2).mean().item()
                        rel   = max(0.0, min(1.0, rel))
                        block_diffs[idx]  = block_diffs.get(idx, 0.0) + rel
                        block_counts[idx] = block_counts.get(idx, 0) + 1

                    del sd_a, sd_b
                    torch.cuda.empty_cache()

                    if not block_diffs:
                        return "❌  No matching keys found — models may be incompatible.", ""

                    # Average per block → difference % (0=identical, 100=very different)
                    results = {}
                    for idx in block_diffs:
                        avg = block_diffs[idx] / block_counts[idx]
                        results[idx] = round(avg * 100, 2)

                    # Color: blue=low diff, yellow=medium, red=high diff
                    def bar_color(pct):
                        if pct <= 5:
                            return "rgb(80,160,220)"     # blue — nearly identical
                        elif pct <= 15:
                            return "rgb(100,200,100)"    # green — small diff
                        elif pct <= 30:
                            return "rgb(240,190,50)"     # yellow — moderate
                        elif pct <= 50:
                            return "rgb(240,120,40)"     # orange — large
                        else:
                            return "rgb(220,60,60)"      # red — very different

                    def diff_label(pct):
                        if pct <= 2:   return "nearly identical"
                        elif pct <= 8: return "very similar"
                        elif pct <= 20: return "somewhat different"
                        elif pct <= 40: return "quite different"
                        else:           return "very different"

                    rows = ""
                    for i, label in enumerate(CKPT_BLOCK_LABELS):
                        pct      = results.get(i, 0.0)
                        col      = bar_color(pct)
                        bar_w    = min(100, pct * 2)  # scale for visibility
                        sem_tags = BLOCK_SEMANTIC_LABELS.get(i, [])
                        sem_html = ""
                        if sem_tags:
                            tags = "".join(
                                f'<span style="display:inline-block;margin:1px 2px;'
                                f'padding:1px 5px;border-radius:3px;font-size:10px;'
                                f'background:var(--color-background-secondary);'
                                f'color:var(--color-text-secondary)">{t}</span>'
                                for t in sem_tags
                            )
                            sem_html = f'<div style="margin-top:2px">{tags}</div>'

                        rows += (
                            f'<tr style="border-bottom:1px solid var(--color-border-tertiary)">'
                            f'<td style="padding:6px 8px 6px 0;font-size:12px;'
                            f'font-family:monospace;color:var(--color-text-primary);'
                            f'white-space:nowrap;vertical-align:top">{label}</td>'
                            f'<td style="width:100%;padding:6px 8px;vertical-align:top">'
                            f'<div style="background:var(--color-background-secondary);'
                            f'border-radius:3px;height:12px;overflow:hidden;margin-bottom:3px">'
                            f'<div style="width:{bar_w:.1f}%;height:100%;'
                            f'background:{col};border-radius:3px;'
                            f'transition:width 0.3s"></div></div>'
                            f'{sem_html}</td>'
                            f'<td style="padding:6px 0 6px 4px;font-size:12px;'
                            f'white-space:nowrap;vertical-align:top;'
                            f'color:var(--color-text-secondary)">'
                            f'<strong style="color:var(--color-text-primary)">{pct:.1f}%</strong>'
                            f' diff</td>'
                            f'<td style="padding:6px 0 6px 6px;font-size:11px;'
                            f'white-space:nowrap;vertical-align:top;'
                            f'color:var(--color-text-secondary)">{diff_label(pct)}</td>'
                            f'</tr>'
                        )

                    avg_all  = sum(results.values()) / len(results) if results else 0
                    max_diff = max(results.values()) if results else 0
                    # Find the most different block
                    most_diff_idx = max(results, key=results.get) if results else 0
                    most_diff_lbl = CKPT_BLOCK_LABELS[most_diff_idx] if most_diff_idx < len(CKPT_BLOCK_LABELS) else "?"
                    most_diff_sem = ", ".join(BLOCK_SEMANTIC_LABELS.get(most_diff_idx, ["unknown"]))

                    summary = (
                        f'<div style="padding:10px 12px;margin-bottom:12px;'
                        f'background:var(--color-background-secondary);border-radius:6px;'
                        f'font-size:13px">'
                        f'<strong>Average difference:</strong> {avg_all:.2f}% &nbsp;·&nbsp; '
                        f'<strong>Most different block:</strong> {most_diff_lbl} '
                        f'({max_diff:.1f}% — {most_diff_sem})'
                        f'</div>'
                    )

                    legend = (
                        f'<div style="display:flex;gap:12px;flex-wrap:wrap;'
                        f'font-size:11px;color:var(--color-text-secondary);margin-bottom:8px">'
                        f'<span><span style="display:inline-block;width:10px;height:10px;'
                        f'border-radius:2px;background:rgb(80,160,220);margin-right:3px"></span>nearly identical</span>'
                        f'<span><span style="display:inline-block;width:10px;height:10px;'
                        f'border-radius:2px;background:rgb(100,200,100);margin-right:3px"></span>small diff</span>'
                        f'<span><span style="display:inline-block;width:10px;height:10px;'
                        f'border-radius:2px;background:rgb(240,190,50);margin-right:3px"></span>moderate</span>'
                        f'<span><span style="display:inline-block;width:10px;height:10px;'
                        f'border-radius:2px;background:rgb(240,120,40);margin-right:3px"></span>large</span>'
                        f'<span><span style="display:inline-block;width:10px;height:10px;'
                        f'border-radius:2px;background:rgb(220,60,60);margin-right:3px"></span>very different</span>'
                        f'</div>'
                    )

                    html = (
                        f'<div style="padding:8px 0">'
                        f'{summary}{legend}'
                        f'<table style="width:100%;border-collapse:collapse">{rows}</table>'
                        f'</div>'
                    )

                    warn = ""
                    if skipped_shape > 0:
                        warn = f" ⚠️ {skipped_shape} keys skipped (shape mismatch — different architectures)"
                    return f"✅  Analysis complete — {len(results)} blocks.{warn}", html

                bm_analyze_btn.click(
                    fn=bm_analyze,
                    inputs=[bm_model_a, bm_model_b],
                    outputs=[bm_analyze_status, bm_diff_map]
                )

                def bm_stop():
                    _STOP_MERGE[0] = True
                    _log("⏹  Stop requested by user...")
                    return "⏹  Stop requested — finishing current operation..."

                bm_stop_btn.click(fn=bm_stop, outputs=[bm_status])

                bm_run_btn.click(
                    fn=bm_run,
                    inputs=[bm_method, bm_alpha, bm_use_blocks, bm_mode,
                             bm_model_a, bm_model_b, bm_model_c,
                             bm_out_name, bm_precision, bm_vae, bm_save_meta] + bm_all,
                    outputs=[bm_status, bm_progress]
                )
                bm_p_save.click(bm_save_p,
                    inputs=[bm_p_name, bm_method, bm_alpha, bm_use_blocks, bm_mode] + bm_all,
                    outputs=[bm_p_status, bm_p_dd])
                bm_p_load.click(bm_load_p,
                    inputs=[bm_p_dd],
                    outputs=[bm_p_status, bm_method, bm_alpha, bm_use_blocks, bm_mode] + bm_all)
                bm_p_del.click(bm_del_p,
                    inputs=[bm_p_dd],
                    outputs=[bm_p_status, bm_p_dd])

            # ═════════════════════════════════════════════════════════════
            #  TAB 2 — LORA TOOLS
            # ═════════════════════════════════════════════════════════════
            with gr.Tab("🔧  LoRA Tools"):

                with gr.Tabs():

                    # ─── Sub-tab A: Bake-in ───────────────────────────
                    with gr.Tab("💉  Bake-in  (LoRA → Checkpoint)"):

                        gr.Markdown(
                            "Permanently fuse a LoRA into a checkpoint. "
                            "The output is a standalone `.safetensors` with the LoRA baked in."
                        )
                        gr.Markdown("")

                        with gr.Row(equal_height=True):

                            with gr.Column(scale=1, min_width=300):
                                with gr.Group():
                                    gr.Markdown("### 📂 Files")
                                    ll = get_lora_list()
                                    ml = get_model_list()
                                    bi_ckpt    = gr.Dropdown(ml, label="Checkpoint")
                                    bi_lora    = gr.Dropdown(ll, label="LoRA")
                                    bi_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                                    bi_refresh.click(
                                        fn=lambda: [gr.update(choices=get_model_list()),
                                                    gr.update(choices=get_lora_list())],
                                        outputs=[bi_ckpt, bi_lora]
                                    )

                                gr.Markdown("")

                                with gr.Group():
                                    gr.Markdown("### 💾 Output")
                                    bi_out_name = gr.Textbox(label="File name", value="baked_model",
                                                             placeholder="e.g. nexus_lora_baked")
                                    gr.Markdown("<small>Saved as `.safetensors` in your checkpoints folder.</small>")

                                gr.Markdown("")

                                with gr.Accordion("⚙️ Advanced Options", open=False):
                                    bi_precision = gr.Radio(
                                        ["fp16", "bf16", "fp32", "fp8"], value="fp16",
                                        label="Output precision"
                                    )
                                    with gr.Row():
                                        bi_vae = gr.Dropdown(
                                            ["— keep original —"] + get_vae_list(),
                                            value="— keep original —",
                                            label="VAE swap", scale=4
                                        )
                                        bi_vae_ref = gr.Button("↻", size="sm", scale=1)
                                        bi_vae_ref.click(
                                            fn=lambda: gr.update(choices=["— keep original —"] + get_vae_list()),
                                            outputs=[bi_vae]
                                        )
                                    bi_save_meta = gr.Checkbox(value=True, label="Save merge metadata")

                                gr.Markdown("")

                                (bi_p_name, bi_p_save, bi_p_dd,
                                 bi_p_load, bi_p_del, bi_p_status) = build_preset_row("bi_")

                                gr.Markdown("")
                                with gr.Row():
                                    bi_run_btn  = gr.Button("💉  Bake LoRA", variant="primary", scale=3)
                                    bi_stop_btn = gr.Button("⏹  Stop",       variant="stop",    scale=1)
                                bi_status   = gr.Markdown("")
                                bi_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress",
                                                        interactive=False)

                            with gr.Column(scale=2):
                                with gr.Group():
                                    gr.Markdown("### ⚙️ Mode")
                                    bi_mode = gr.Radio(
                                        ["Easy", "Advanced"], value="Easy", label="",
                                        info="Easy = single strength slider  |  Advanced = per-category control"
                                    )

                                gr.Markdown("")

                                # Easy — single strength
                                with gr.Group(visible=True) as bi_easy_panel:
                                    gr.Markdown("### 🎛️ LoRA Strength")
                                    gr.Markdown(
                                        "<small>Controls how strongly the LoRA is baked into the checkpoint. "
                                        "**1.0** = full strength (same as running the LoRA at 1.0 during inference).</small>"
                                    )
                                    gr.Markdown("")
                                    bi_strength = gr.Slider(
                                        0.0, 2.0, value=1.0, step=0.05,
                                        label="Strength"
                                    )
                                    gr.Markdown("")
                                    gr.Markdown(
                                        "<small>💡 **Tip:** Values above 1.0 amplify the LoRA effect. "
                                        "Most LoRAs work best between 0.6 and 1.0.</small>"
                                    )

                                # Advanced — semantic sliders
                                with gr.Group(visible=False) as bi_adv_panel:
                                    gr.Markdown("### 🎛️ Per-Category Strength")
                                    gr.Markdown(
                                        "<small>Control how strongly the LoRA is baked in for each semantic area. "
                                        "**0** = no effect &nbsp;·&nbsp; **1** = full strength &nbsp;·&nbsp; "
                                        "**2** = amplified</small>"
                                    )
                                    gr.Markdown("")
                                    bi_adv_sliders = {}

                                    # Standard sliders
                                    for cat_name, cat_info in EASY_CATEGORIES.items():
                                        if cat_name.startswith("⚗️ [EXP]"):
                                            continue
                                        with gr.Row():
                                            gr.Markdown(
                                                f"**{cat_info['icon']} {cat_name}**  "
                                                f"<small style='color:gray'>{cat_info['description']}</small>",
                                                scale=2
                                            )
                                            bi_adv_sliders[cat_name] = gr.Slider(
                                                0.0, 2.0, value=1.0, step=0.05,
                                                label="", show_label=False, scale=3,
                                            )

                                    # Experimental sliders — collapsed
                                    with gr.Accordion("⚗️ Experimental sliders", open=False):
                                        gr.Markdown(
                                            "<small style='color:orange'>Results are unpredictable and vary between models.</small>"
                                        )
                                        gr.Markdown("")
                                        for cat_name, cat_info in EASY_CATEGORIES.items():
                                            if not cat_name.startswith("⚗️ [EXP]"):
                                                continue
                                            label = cat_name.replace("⚗️ [EXP] ", "")
                                            with gr.Row():
                                                gr.Markdown(
                                                    f"**{cat_info['icon']} {label}**  "
                                                    f"<small style='color:gray'>{cat_info['description'].split(' — ')[0]}</small>",
                                                    scale=2
                                                )
                                                bi_adv_sliders[cat_name] = gr.Slider(
                                                    0.0, 2.0, value=1.0, step=0.05,
                                                    label="", show_label=False, scale=3,
                                                )

                                bi_mode.change(
                                    fn=lambda m: (gr.update(visible=m=="Easy"),
                                                  gr.update(visible=m=="Advanced")),
                                    inputs=bi_mode,
                                    outputs=[bi_easy_panel, bi_adv_panel]
                                )

                        bi_all = [bi_strength] + list(bi_adv_sliders.values())

                        def bi_run(mode, ckpt, lora, out_n, precision, vae, save_meta, strength, *adv_vals):
                            if not ckpt or not lora:
                                return "❌  Select a Checkpoint and a LoRA.", 0.0
                            strength  = float(strength) if strength is not None else 1.0
                            adv_vals  = tuple(float(v) if v is not None else 0.0 for v in adv_vals)
                            if mode == "Easy":
                                bw = [strength] * 20
                            else:
                                adv_dict = dict(zip(bi_adv_sliders.keys(), adv_vals))
                                contributions = [0.0] * 20
                                counts        = [0]   * 20
                                for cat_name, val in adv_dict.items():
                                    if cat_name not in EASY_CATEGORIES:
                                        continue
                                    for block_name, base_w in EASY_CATEGORIES[cat_name]["blocks"].items():
                                        if block_name not in CKPT_BLOCK_KEYS:
                                            continue
                                        idx = CKPT_BLOCK_KEYS.index(block_name)
                                        contributions[idx] += base_w * val
                                        counts[idx]        += 1
                                bw = [
                                    min(2.0, contributions[i] / counts[i]) if counts[i] > 0 else 1.0
                                    for i in range(20)
                                ]
                            bw = [float(v) for v in bw]
                            pc  = find_model(ckpt)
                            pl  = find_lora(lora)
                            vp  = find_vae(vae) if vae and vae != "— keep original —" else None
                            op  = os.path.join(get_ckpt_dir(),
                                              (out_n.strip().replace(" ","_") or "baked_model") + ".safetensors")
                            try:
                                fused = bake_lora_into_checkpoint(pc, pl, bw, op,
                                                                   precision=precision,
                                                                   vae_path=vp,
                                                                   save_metadata=save_meta)
                                return f"✅  Baked {fused} layers → `{op}`", 1.0
                            except Exception as e:
                                return f"❌  {e}", 0.0

                        def bi_save_p(name, mode, strength, *adv_vals):
                            if not name.strip(): return "❌  Enter a name.", gr.update()
                            save_preset(name.strip(), {
                                "mode": mode,
                                "strength": strength,
                                "advanced": dict(zip(bi_adv_sliders.keys(), adv_vals)),
                            }, "bi_")
                            return f"✅  Saved **{name}**.", gr.update(choices=list_presets("bi_"))

                        def bi_load_p(name):
                            d = load_preset(name, "bi_")
                            if not d:
                                return (["❌  Not found."]
                                        + [gr.update()] * (len(bi_adv_sliders) + 2))
                            av = [d.get("advanced", {}).get(k, 1.0) for k in bi_adv_sliders.keys()]
                            return ([f"✅  Loaded **{name}**.",
                                     gr.update(value=d.get("mode", "Easy")),
                                     gr.update(value=d.get("strength", 1.0))]
                                    + [gr.update(value=v) for v in av])

                        def bi_del_p(name):
                            delete_preset(name, "bi_")
                            return f"🗑️  Deleted **{name}**.", gr.update(choices=list_presets("bi_"))

                        def bi_stop():
                            _STOP_MERGE[0] = True
                            _log("⏹  Stop requested by user...")
                            return "⏹  Stop requested — finishing current operation..."

                        bi_stop_btn.click(fn=bi_stop, outputs=[bi_status])

                        bi_run_btn.click(bi_run,
                            inputs=[bi_mode, bi_ckpt, bi_lora, bi_out_name,
                                     bi_precision, bi_vae, bi_save_meta] + bi_all,
                            outputs=[bi_status, bi_progress])
                        bi_p_save.click(bi_save_p,
                            inputs=[bi_p_name, bi_mode] + bi_all,
                            outputs=[bi_p_status, bi_p_dd])
                        bi_p_load.click(bi_load_p,
                            inputs=[bi_p_dd],
                            outputs=[bi_p_status, bi_mode] + bi_all)
                        bi_p_del.click(bi_del_p,
                            inputs=[bi_p_dd],
                            outputs=[bi_p_status, bi_p_dd])

                    # ─── Sub-tab B: LoRA Merge ────────────────────────
                    with gr.Tab("🔗  LoRA Merge  (LoRA + LoRA)"):

                        gr.Markdown(
                            "Combine two LoRAs into a single new LoRA file. "
                            "Use the slider to blend between them."
                        )
                        gr.Markdown("")

                        with gr.Row(equal_height=True):

                            with gr.Column(scale=1, min_width=300):
                                with gr.Group():
                                    gr.Markdown("### 📂 LoRA Files")
                                    ll2 = get_lora_list()
                                    lm_lora_a  = gr.Dropdown(ll2, label="LoRA A")
                                    lm_lora_b  = gr.Dropdown(ll2, label="LoRA B")
                                    lm_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                                    lm_refresh.click(
                                        fn=lambda: [gr.update(choices=get_lora_list())] * 2,
                                        outputs=[lm_lora_a, lm_lora_b]
                                    )

                                gr.Markdown("")

                                with gr.Group():
                                    gr.Markdown("### 💾 Output")
                                    lm_out_name = gr.Textbox(label="File name", value="merged_lora",
                                                             placeholder="e.g. style_anatomy_mix")
                                    gr.Markdown("<small>Saved as `.safetensors` in your LoRA folder.</small>")

                                gr.Markdown("")

                                with gr.Accordion("⚙️ Advanced Options", open=False):
                                    lm_precision = gr.Radio(
                                        ["fp16", "bf16", "fp32", "fp8"], value="fp16",
                                        label="Output precision"
                                    )
                                    lm_save_meta = gr.Checkbox(value=True, label="Save merge metadata")

                                gr.Markdown("")
                                with gr.Row():
                                    lm_run_btn  = gr.Button("🔗  Merge LoRAs", variant="primary", scale=3)
                                    lm_stop_btn = gr.Button("⏹  Stop",         variant="stop",    scale=1)
                                lm_status   = gr.Markdown("")
                                lm_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress",
                                                        interactive=False)

                            with gr.Column(scale=2):
                                with gr.Group():
                                    gr.Markdown("### ⚖️ Blend")
                                    gr.Markdown(
                                        "<small>**0** = 100% LoRA B &nbsp;·&nbsp; "
                                        "**0.5** = equal mix &nbsp;·&nbsp; "
                                        "**1** = 100% LoRA A</small>"
                                    )
                                    gr.Markdown("")
                                    lm_weight = gr.Slider(
                                        0.0, 1.0, value=0.5, step=0.05,
                                        label="LoRA A weight"
                                    )
                                    gr.Markdown("")
                                    gr.Markdown(
                                        "### ℹ️ Notes\n"
                                        "- LoRAs with **different ranks** are supported — "
                                        "the smaller one is zero-padded to match.\n"
                                        "- The output rank equals the **larger** of the two LoRAs.\n"
                                        "- For best results, merge LoRAs trained on the same base model."
                                    )

                        def lm_run(la, lb, weight, out_n, precision, save_meta):
                            if not la or not lb:
                                return "❌  Select both LoRA A and LoRA B.", 0.0
                            pa  = find_lora(la)
                            pb  = find_lora(lb)
                            op  = os.path.join(get_lora_dir(),
                                               (out_n.strip().replace(" ","_") or "merged_lora") + ".safetensors")
                            try:
                                merge_loras(pa, pb, weight, op,
                                            precision=precision,
                                            save_metadata=save_meta)
                                return f"✅  Saved to `{op}`", 1.0
                            except Exception as e:
                                return f"❌  {e}", 0.0

                        def lm_stop():
                            _STOP_MERGE[0] = True
                            _log("⏹  Stop requested by user...")
                            return "⏹  Stop requested — finishing current operation..."

                        lm_stop_btn.click(fn=lm_stop, outputs=[lm_status])

                        lm_run_btn.click(lm_run,
                            inputs=[lm_lora_a, lm_lora_b, lm_weight, lm_out_name,
                                     lm_precision, lm_save_meta],
                            outputs=[lm_status, lm_progress])

            # ═════════════════════════════════════════════════════════════
            #  TAB 3 — INSPECT
            # ═════════════════════════════════════════════════════════════
            with gr.Tab("🔍  Inspect"):

                gr.Markdown(
                    "Load any `.safetensors` or `.ckpt` file and inspect its contents — "
                    "architecture, precision, file size, keys, and any embedded metadata "
                    "(including NeoMerger merge recipes)."
                )
                gr.Markdown("")

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=280):
                        with gr.Group():
                            gr.Markdown("### 📂 File")
                            all_models = get_model_list() + [f"[LoRA] {l}" for l in get_lora_list()]
                            ins_file    = gr.Dropdown(all_models, label="Select model or LoRA")
                            ins_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                            ins_refresh.click(
                                fn=lambda: gr.update(
                                    choices=get_model_list() + [f"[LoRA] {l}" for l in get_lora_list()]
                                ),
                                outputs=[ins_file]
                            )
                            ins_btn = gr.Button("🔍  Inspect", variant="primary")

                        gr.Markdown("")
                        ins_status = gr.Markdown("")

                    with gr.Column(scale=2):

                        with gr.Group():
                            gr.Markdown("### 📋 General info")
                            ins_general = gr.Markdown("<small>Select a file and click Inspect.</small>")

                        gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 🔀 Merge recipe")
                            gr.Markdown("<small>Shown if the file was created with NeoMerger.</small>")
                            ins_recipe = gr.Markdown("<small>—</small>")

                        gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 🏷️ All metadata")
                            ins_meta = gr.Markdown("<small>—</small>")

                        gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 🗝️ Key statistics")
                            ins_keys = gr.Markdown("<small>—</small>")

                def do_inspect(name):
                    if not name:
                        return "❌  Select a file.", "<small>—</small>", "<small>—</small>", "<small>—</small>"

                    # Resolve path
                    is_lora = name.startswith("[LoRA] ")
                    clean   = name.replace("[LoRA] ", "")
                    path    = find_lora(clean) if is_lora else find_model(clean)

                    if not os.path.isfile(path):
                        return f"❌  File not found: `{path}`", "<small>—</small>", "<small>—</small>", "<small>—</small>"

                    # ── File size ──────────────────────────────────────────
                    size_bytes = os.path.getsize(path)
                    if size_bytes > 1e9:
                        size_str = f"{size_bytes/1e9:.2f} GB"
                    else:
                        size_str = f"{size_bytes/1e6:.1f} MB"

                    # ── Hash ───────────────────────────────────────────────
                    try:
                        file_hash = compute_hash(path)
                    except Exception:
                        file_hash = "N/A"

                    # ── Load metadata + keys ───────────────────────────────
                    raw_meta   = {}
                    key_dtypes = {}
                    total_keys = 0
                    total_params = 0

                    try:
                        if path.endswith(".safetensors"):
                            from safetensors import safe_open
                            with safe_open(path, framework="pt", device="cpu") as sf:
                                raw_meta   = dict(sf.metadata() or {})
                                key_list   = sf.keys()
                                total_keys = len(list(key_list))
                                for k in sf.keys():
                                    t = sf.get_tensor(k)
                                    dt = str(t.dtype).replace("torch.", "")
                                    key_dtypes[dt] = key_dtypes.get(dt, 0) + 1
                                    total_params   += t.numel()
                        else:
                            sd = load_sd(path)
                            total_keys = len(sd)
                            for v in sd.values():
                                dt = str(v.dtype).replace("torch.", "")
                                key_dtypes[dt] = key_dtypes.get(dt, 0) + 1
                                total_params   += v.numel()
                    except Exception as e:
                        return f"❌  Could not read file: {e}", "<small>—</small>", "<small>—</small>", "<small>—</small>"

                    # ── Detect architecture ────────────────────────────────
                    sd_tmp = load_sd(path) if not path.endswith(".safetensors") else None
                    try:
                        if path.endswith(".safetensors"):
                            from safetensors import safe_open
                            with safe_open(path, framework="pt", device="cpu") as sf:
                                all_keys    = list(sf.keys())
                                sample_keys = all_keys[:120]
                        else:
                            all_keys    = list(sd_tmp.keys())
                            sample_keys = all_keys[:120]
                        if sd_tmp: del sd_tmp
                    except Exception:
                        sample_keys = []
                        all_keys    = []

                    meta_arch = (raw_meta.get("modelspec.architecture","") or
                                 raw_meta.get("architecture","") or
                                 raw_meta.get("ss_base_model_version","") or "")

                    def has(patterns, keys):
                        return any(any(p in k for p in patterns) for k in keys)

                    # Flux variants
                    if has(["double_blocks","single_blocks","transformer_blocks.0.attn.norm"], sample_keys):
                        if has(["flux-schnell","schnell"], [meta_arch.lower()]):
                            arch = "Flux.1 Schnell"
                        elif has(["flux-dev","flux.1-dev"], [meta_arch.lower()]):
                            arch = "Flux.1 Dev"
                        else:
                            arch = "Flux.1"

                    # SD3 / SD3.5
                    elif has(["joint_transformer_blocks","adaln_single"], sample_keys):
                        if has(["sd3.5","sd-3.5"], [meta_arch.lower()]):
                            arch = "SD 3.5"
                        else:
                            arch = "SD 3.x"

                    # SDXL family
                    elif has(["conditioner.embedders","add_embedding","label_emb"], sample_keys):
                        # Try to distinguish Pony / Illustrious from base SDXL via metadata
                        m = meta_arch.lower()
                        if "pony" in m or "pdxl" in m:
                            arch = "Pony Diffusion XL"
                        elif "illustrious" in m or "noob" in m:
                            arch = "Illustrious XL"
                        elif "playground" in m:
                            arch = "Playground v2.5"
                        else:
                            arch = "SDXL 1.0"

                    # SD 2.x
                    elif has(["model.diffusion_model"], sample_keys) and has(["attn2.to_k","time_embed"], sample_keys):
                        # SD2 uses OpenCLIP (no cond_stage_model.transformer)
                        if not has(["cond_stage_model.transformer"], sample_keys):
                            arch = "SD 2.x"
                        else:
                            arch = "SD 1.x"

                    # SD 1.x (most common)
                    elif has(["model.diffusion_model","first_stage_model","cond_stage_model"], sample_keys):
                        # Count input blocks to distinguish 1.4/1.5/1.5-inpainting
                        n_in = sum(1 for k in sample_keys if "input_blocks" in k)
                        if has(["inpaint"], [path.lower()]):
                            arch = "SD 1.5 (Inpainting)"
                        elif n_in > 0:
                            arch = "SD 1.x  (SD 1.4 / 1.5)"
                        else:
                            arch = "SD 1.x"

                    # LoRA files
                    elif has(["lora_unet","lora_te","lora_te1","lora_te2"], sample_keys):
                        if has(["lora_te2","lora_unet_down_blocks_2"], sample_keys):
                            arch = "LoRA — SDXL"
                        elif has(["lora_unet_double","lora_unet_single"], sample_keys):
                            arch = "LoRA — Flux"
                        else:
                            arch = "LoRA — SD 1.x"

                    # LyCORIS / other LoRA formats
                    elif has(["oft_blocks","hada_w1","lokr_w1","glora"], sample_keys):
                        arch = "LyCORIS"

                    # VAE standalone
                    elif has(["encoder.down","decoder.up","quant_conv"], sample_keys) and not has(["model.diffusion_model"], sample_keys):
                        arch = "VAE (standalone)"

                    # Textual Inversion embedding
                    elif has(["emb_params","string_to_param"], sample_keys):
                        arch = "Textual Inversion Embedding"

                    else:
                        # Last resort: check total key count heuristic
                        n = len(all_keys)
                        if n < 50:
                            arch = f"Unknown (few keys: {n})"
                        elif n < 500:
                            arch = "Unknown — possibly LoRA or embedding"
                        else:
                            arch = f"Unknown ({n:,} keys)"

                    dtype_str  = ", ".join(f"{dt}: {n}" for dt, n in sorted(key_dtypes.items()))
                    params_str = f"{total_params/1e9:.2f}B" if total_params > 1e9 else f"{total_params/1e6:.0f}M"

                    nl = "  \n"
                    general = (
                        f"**File:** `{os.path.basename(path)}`" + nl +
                        f"**Size:** {size_str}" + nl +
                        f"**Hash (SHA256):** `{file_hash}`" + nl +
                        f"**Architecture:** {arch}" + nl +
                        f"**Total keys:** {total_keys:,}" + nl +
                        f"**Total parameters:** {params_str}" + nl +
                        f"**Precisions found:** {dtype_str or 'N/A'}"
                    )

                    # ── Merge recipe ──────────────────────────────────────
                    recipe_lines = []

                    # 1. NeoMerger metadata
                    if raw_meta and "neomerger_version" in raw_meta:
                        mt = raw_meta.get("merge_type", "unknown")
                        recipe_lines.append(f"**🔀 NeoMerger merge** (v{raw_meta.get('neomerger_version','?')})")
                        recipe_lines.append(f"**Type:** {mt}")
                        if mt == "block_merge":
                            recipe_lines += [
                                f"**Model A:** `{raw_meta.get('model_a','?')}`",
                                f"**Model B:** `{raw_meta.get('model_b','?')}`",
                                f"**Precision:** {raw_meta.get('precision','?')}",
                                f"**VAE:** {raw_meta.get('vae','original')}",
                            ]
                            bw_raw = raw_meta.get("block_weights")
                            if bw_raw:
                                try:
                                    bw = json.loads(bw_raw)
                                    bw_fmt = " ".join(
                                        f"`{CKPT_BLOCK_LABELS[i]}={v:.2f}`"
                                        for i, v in enumerate(bw)
                                    )
                                    recipe_lines.append("**Block weights:**  \n" + bw_fmt)
                                except Exception:
                                    recipe_lines.append(f"**Block weights:** `{bw_raw}`")
                        elif mt == "lora_bake":
                            recipe_lines += [
                                f"**Checkpoint:** `{raw_meta.get('checkpoint','?')}`",
                                f"**LoRA:** `{raw_meta.get('lora','?')}`",
                                f"**Fused layers:** {raw_meta.get('fused_layers','?')}",
                                f"**Precision:** {raw_meta.get('precision','?')}",
                            ]
                        elif mt == "lora_merge":
                            recipe_lines += [
                                f"**LoRA A:** `{raw_meta.get('lora_a','?')}` — weight {raw_meta.get('weight_a','?')}",
                                f"**LoRA B:** `{raw_meta.get('lora_b','?')}` — weight {raw_meta.get('weight_b','?')}",
                                f"**Precision:** {raw_meta.get('precision','?')}",
                            ]

                    # 2. Supermerger metadata
                    elif raw_meta and any(k in raw_meta for k in ["sd_merge_models","sd_merge_recipe"]):
                        recipe_lines.append("**🔀 Merged with Supermerger**")
                        merge_recipe = raw_meta.get("sd_merge_recipe","")
                        merge_models = raw_meta.get("sd_merge_models","")
                        if merge_models:
                            recipe_lines.append(f"**Models:** {merge_models}")
                        if merge_recipe:
                            recipe_lines.append(f"**Recipe:** `{merge_recipe}`")

                    # 3. modelspec merge info (used by some tools)
                    elif raw_meta and "modelspec.title" in raw_meta:
                        recipe_lines.append("**📋 modelspec metadata**")
                        for k in ["modelspec.title","modelspec.author","modelspec.description",
                                  "modelspec.architecture","modelspec.date","modelspec.license"]:
                            if k in raw_meta:
                                label = k.replace("modelspec.","").capitalize()
                                recipe_lines.append(f"**{label}:** {raw_meta[k]}")

                    # 4. LoRA training metadata (kohya_ss / other trainers)
                    elif raw_meta and any(k.startswith("ss_") for k in raw_meta):
                        recipe_lines.append("**🎓 LoRA training info**")
                        lora_fields = {
                            "ss_base_model_version": "Base model",
                            "ss_network_module":     "Network type",
                            "ss_network_dim":        "Network dim (rank)",
                            "ss_network_alpha":      "Network alpha",
                            "ss_num_train_images":   "Training images",
                            "ss_num_epochs":         "Epochs",
                            "ss_learning_rate":      "Learning rate",
                            "ss_lr_scheduler":       "LR scheduler",
                            "ss_optimizer":          "Optimizer",
                            "ss_mixed_precision":    "Mixed precision",
                            "ss_resolution":         "Resolution",
                            "ss_tag_frequency":      "Tag frequency",
                        }
                        for key, label in lora_fields.items():
                            if key in raw_meta:
                                val = raw_meta[key]
                                # Truncate very long values
                                if len(str(val)) > 120:
                                    val = str(val)[:120] + "..."
                                recipe_lines.append(f"**{label}:** {val}")

                    if raw_meta:
                        for k in ["trigger_words","activation text","ss_tag_frequency"]:
                            if k in raw_meta and k != "ss_tag_frequency":
                                recipe_lines.append(f"**Trigger words:** `{raw_meta[k]}`")
                                break

                    recipe = "  \n".join(recipe_lines) if recipe_lines else "<small>No merge recipe or training metadata found in this file.</small>"

                    # ── All metadata ───────────────────────────────────────
                    if raw_meta:
                        meta_lines = [f"**`{k}`:** {v}" for k, v in sorted(raw_meta.items())]
                        meta_out = "  \n".join(meta_lines)
                    else:
                        meta_out = "<small>No metadata found in this file.</small>"

                    # ── Key statistics ─────────────────────────────────────
                    unet_keys  = sum(1 for k in sample_keys if "model.diffusion_model" in k or "lora_unet" in k)
                    te_keys    = sum(1 for k in sample_keys if "cond_stage" in k or "conditioner" in k or "lora_te" in k)
                    vae_keys   = sum(1 for k in sample_keys if "first_stage_model" in k)

                    keys_out = (
                        f"**UNet / diffusion keys:** ~{unet_keys}+ (from first 80 sampled)" + nl +
                        f"**Text encoder keys:** ~{te_keys}+" + nl +
                        f"**VAE keys:** ~{vae_keys}+" + nl +
                        "*Full key count above is exact.*"
                    )

                    return "✅  Inspection complete.", general, recipe, meta_out

                ins_btn.click(
                    fn=do_inspect,
                    inputs=[ins_file],
                    outputs=[ins_status, ins_general, ins_recipe, ins_meta]
                )

                gr.Markdown("")

                with gr.Group():
                    gr.Markdown("### 🗑️ Clear metadata")
                    gr.Markdown(
                        "<small>⚠️ **Warning:** This rewrites the entire file without metadata. "
                        "For a 7 GB SDXL model it takes the same time as a normal merge. "
                        "The original file will be overwritten — make a backup first.</small>"
                    )
                    gr.Markdown("")
                    with gr.Row():
                        ins_clear_file = gr.Dropdown(
                            get_model_list() + [f"[LoRA] {l}" for l in get_lora_list()],
                            label="Select file to clear", scale=4
                        )
                        ins_clear_ref = gr.Button("↻", size="sm", scale=1)
                        ins_clear_ref.click(
                            fn=lambda: gr.update(
                                choices=get_model_list() + [f"[LoRA] {l}" for l in get_lora_list()]
                            ),
                            outputs=[ins_clear_file]
                        )
                    ins_clear_btn    = gr.Button("🗑️  Clear metadata", variant="stop")
                    ins_clear_status = gr.Markdown("")
                    ins_clear_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress", interactive=False)

                def do_clear_metadata(name):
                    if not name:
                        return "❌  Select a file.", 0.0

                    is_lora = name.startswith("[LoRA] ")
                    clean   = name.replace("[LoRA] ", "")
                    path    = find_lora(clean) if is_lora else find_model(clean)

                    if not os.path.isfile(path):
                        return f"❌  File not found: `{path}`", 0.0

                    if not path.endswith(".safetensors"):
                        return "❌  Only .safetensors files are supported.", 0.0

                    tmp_path = path + ".tmp_clear"

                    try:
                        from safetensors import safe_open
                        from safetensors.torch import save_file

                        tensors = {}
                        with safe_open(path, framework="pt", device="cpu") as sf:
                            for key in sf.keys():
                                tensors[key] = sf.get_tensor(key)

                        save_file(tensors, tmp_path)
                        del tensors
                        torch.cuda.empty_cache()

                        os.replace(tmp_path, path)

                        size = os.path.getsize(path) / 1e9
                        return f"✅  Metadata cleared from `{os.path.basename(path)}` ({size:.2f} GB)", 1.0

                    except Exception as e:
                        if os.path.isfile(tmp_path):
                            try: os.remove(tmp_path)
                            except Exception: pass
                        return f"❌  Error: {e} — original file is unchanged.", 0.0

                ins_clear_btn.click(
                    fn=do_clear_metadata,
                    inputs=[ins_clear_file],
                    outputs=[ins_clear_status, ins_clear_progress]
                )

    return [(ui, "NeoMerger", "neo_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)
