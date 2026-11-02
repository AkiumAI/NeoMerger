"""
NeoMerger — Advanced model merging extension
Developed by AkiumAI — https://github.com/AkiumAI

Supports: Forge Neo, Forge, reForge, AUTOMATIC1111
Tabs: Merge · LoRA Tools (Bake-in · LoRA Merge · Extract) · Inspect · Block Probe

License: CC BY-NC 4.0
https://creativecommons.org/licenses/by-nc/4.0/
© 2025 AkiumAI
"""

import os
import json
import torch
import gradio as gr
from pathlib import Path

from modules import script_callbacks, shared, sd_models, processing
from modules.paths import models_path, data_path

NEOMERGER_VERSION = "1.5.0"

# Global stop flag: set to True to interrupt the running operation.
_STOP_MERGE = [False]

# Seconds spent in _quantile_safe during a merge, reported in the terminal.
_QUANTILE_TIME = [0.0]

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


# ── Output file names ────────────────────────────────────────────────────────
#
# Controlled by two settings (Settings → NeoMerger). Input models are never
# overwritten, whatever the settings: they are still being read while the
# output is written.

OPT_AUTO_NAME  = "neomerger_auto_filename"
OPT_OVERWRITE  = "neomerger_overwrite"
AUTO_NAME_EMPTY  = "When the name field is empty"
AUTO_NAME_ALWAYS = "Always (ignore the name field)"
OVERWRITE_KEEP   = "Keep both — add _2, _3, … to the new file"
OVERWRITE_YES    = "Overwrite the existing file"

_NAME_PLACEHOLDER = "Leave empty for an automatic name"

_INVALID_FN_CHARS = set('<>:"/\\|?*')


def _opt(key, default):
    try:
        v = getattr(shared.opts, key, default)
        return v if isinstance(v, str) else default
    except Exception:
        return default


def _clean_filename(name: str, limit: int = 150) -> str:
    """Make a string safe as a Windows/Linux file name stem."""
    import re as _re
    name = (name or "").strip()
    if name.lower().endswith(".safetensors"):
        name = name[:-len(".safetensors")]
    name = "".join("_" if (c in _INVALID_FN_CHARS or ord(c) < 32) else c for c in name)
    name = _re.sub(r"\s+", "_", name)
    name = _re.sub(r"_{2,}", "_", name).strip(" ._")
    return name[:limit]


def _name_stem(model_name, limit: int = 40) -> str:
    """Short, file-safe stem of a model/LoRA name or path for auto names."""
    base = os.path.splitext(os.path.basename(str(model_name or "")))[0]
    return _clean_filename(base, limit) or "model"


def _num(x) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "x"


def _method_slug(method) -> str:
    import re as _re
    m = str(method or "merge").split("(")[0].strip().lower()
    return _re.sub(r"[^a-z0-9]+", "-", m).strip("-") or "merge"


def auto_merge_name(method, alpha, ma, mb, mc=None, extra=None, block_weights=False,
                    suffix=None):
    """e.g. ModelA_x_ModelB_weighted-sum_a0.50  (· _x_ModelC · _+2 · _bw)
    suffix replaces the alpha/bw tag (the Alpha Sweep uses 'sweep')."""
    parts = [_name_stem(ma), "x", _name_stem(mb)]
    if mc:
        parts += ["x", _name_stem(mc)]
    n_extra = len([e for e in (extra or []) if e])
    if n_extra:
        parts.append(f"+{n_extra}")
    tag = suffix or ("bw" if block_weights else f"a{_num(alpha)}")
    parts += [_method_slug(method), tag]
    return "_".join(parts)


def resolve_output_path(directory: str, typed_name: str, auto_name: str,
                        inputs=()) -> str:
    """
    Final output path for a save, applying the NeoMerger file-name settings.

    typed_name : what the user wrote in the File name box (may be empty)
    auto_name  : name to use when the field is empty / auto mode is 'Always'
    inputs     : paths of the files being read; never overwritten
    """
    mode = _opt(OPT_AUTO_NAME, AUTO_NAME_EMPTY)
    name = "" if mode == AUTO_NAME_ALWAYS else _clean_filename(typed_name)
    if not name:
        name = _clean_filename(auto_name) or "neomerger_output"
    overwrite = _opt(OPT_OVERWRITE, OVERWRITE_KEEP) == OVERWRITE_YES

    def _norm(pth):
        return os.path.normcase(os.path.abspath(pth))
    protected = {_norm(pth) for pth in (inputs or ()) if pth}

    def _taken(pth):
        if not os.path.exists(pth):
            return False
        return (not overwrite) or (_norm(pth) in protected)

    path = os.path.join(directory, name + ".safetensors")
    if _taken(path):
        i = 2
        while _taken(os.path.join(directory, f"{name}_{i}.safetensors")):
            i += 1
        new = os.path.join(directory, f"{name}_{i}.safetensors")
        why = "is one of the input models" if _norm(path) in protected else "already exists"
        _log(f"'{os.path.basename(path)}' {why} — saving as '{os.path.basename(new)}'.")
        path = new
    elif os.path.exists(path):
        _log(f"Overwriting existing '{os.path.basename(path)}' (Settings → NeoMerger).")
    return path


def _on_ui_settings():
    section = ("neomerger", "NeoMerger")
    shared.opts.add_option(OPT_AUTO_NAME, shared.OptionInfo(
        AUTO_NAME_EMPTY, "Automatic output file names", gr.Radio,
        {"choices": [AUTO_NAME_EMPTY, AUTO_NAME_ALWAYS]}, section=section))
    shared.opts.add_option(OPT_OVERWRITE, shared.OptionInfo(
        OVERWRITE_KEEP, "When the output file already exists", gr.Radio,
        {"choices": [OVERWRITE_KEEP, OVERWRITE_YES]}, section=section))


script_callbacks.on_ui_settings(_on_ui_settings)

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

ANIMA_NUM_BLOCKS = 28
ANIMA_BLOCK_LABELS = [f"BLK_{i:02d}" for i in range(ANIMA_NUM_BLOCKS)]

# Depth-upscaled Anima variants exist (2.9B: 40 blocks, 3.8B: 52).
# ANIMA_NUM_BLOCKS is the base depth and only a default: code that indexes
# block weights must use the checkpoint's real depth.
ANIMA_MAX_BLOCKS = 64          # UI slider ceiling; only the detected ones show

_ANIMA_DEPTH_CACHE = {}


def anima_block_count(path_or_sd) -> int:
    """Real number of DiT blocks in an Anima checkpoint (path or state dict).

    Returns ANIMA_NUM_BLOCKS when the depth can't be determined.
    """
    import re as _re
    if not isinstance(path_or_sd, str):
        try:
            keys = list(path_or_sd.keys())
        except Exception:
            return ANIMA_NUM_BLOCKS
        return _anima_depth_from_keys(keys)

    path = path_or_sd
    try:
        mt = os.path.getmtime(path)
    except Exception:
        mt = 0
    ck = (path, mt)
    if ck in _ANIMA_DEPTH_CACHE:
        return _ANIMA_DEPTH_CACHE[ck]

    n = ANIMA_NUM_BLOCKS
    try:
        if path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                n = _anima_depth_from_keys(list(sf.keys()))
        else:
            n = _anima_depth_from_keys(list(load_sd(path).keys()))
    except Exception:
        pass
    _ANIMA_DEPTH_CACHE[ck] = n
    return n


_ANIMA_BLOCK_RE = None

# DiTs driven by the "Anima" per-block sliders. Main stack: blocks.N on Anima
# and Krea 2, layers.N on MOSS-Image (its text_embedder.blocks.N is separate).
DIT_ARCHS = ("anima", "krea2", "moss")
DIT_ARCH_NAMES = {"anima": "Anima", "krea2": "Krea 2", "moss": "MOSS-Image"}

_DIT_TOP_PREFIXES = ("", "net", "diffusion_model", "model.diffusion_model")
_MOSS_LAYER_RE = None


def is_dit_arch(arch) -> bool:
    return str(arch).lower() in DIT_ARCHS or str(arch).lower() == "dit"


def _is_moss_keys(keys) -> bool:
    return any(k.endswith("shared_block_modulation.weight") for k in keys)


def _dit_block_re(keys=None, arch=None):
    """Block-number regex for the main stack: layers.N for MOSS-Image, blocks.N otherwise."""
    global _ANIMA_BLOCK_RE, _MOSS_LAYER_RE
    import re as _re
    if _ANIMA_BLOCK_RE is None:
        _ANIMA_BLOCK_RE = _re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    if _MOSS_LAYER_RE is None:
        _MOSS_LAYER_RE = _re.compile(r"(?:^|\.)layers\.(\d+)\.")
    if arch == "moss" or (arch is None and keys is not None and _is_moss_keys(keys)):
        return _MOSS_LAYER_RE
    return _ANIMA_BLOCK_RE


_DTYPE_CACHE = {}


def detect_checkpoint_dtype(path):
    """Dominant storage dtype of a checkpoint ('bf16', 'fp16', 'fp32', ...).

    Reads the safetensors header only. Used to warn before a lossy bf16 -> fp16 save.
    """
    if not path:
        return None
    try:
        mt = os.path.getmtime(path)
    except Exception:
        mt = 0
    ck = (path, mt)
    if ck in _DTYPE_CACHE:
        return _DTYPE_CACHE[ck]
    out = None
    try:
        if str(path).endswith(".safetensors"):
            from safetensors import safe_open
            from collections import Counter
            with safe_open(path, framework="pt", device="cpu") as f:
                c = Counter()
                for k in f.keys():
                    c[str(f.get_slice(k).get_dtype()).upper()] += 1
                if c:
                    top = c.most_common(1)[0][0]
                    out = {"BF16": "bf16", "F16": "fp16", "F32": "fp32",
                           "F8_E4M3": "fp8", "F8_E5M2": "fp8"}.get(top)
    except Exception:
        out = None
    _DTYPE_CACHE[ck] = out
    return out


def anima_block_namespace(keys) -> str:
    """Prefix in front of `blocks.N.` for the main DiT stack.

    Anima checkpoints also contain the text-encoder adapter, which has its own,
    narrower `blocks.N.` (1024-dim vs the DiT's 2048). Matching on the block
    number alone mixes the two, so pick the namespace with the most keys.
    """
    from collections import Counter
    keys = list(keys)
    rex = _dit_block_re(keys)
    counts = Counter()
    for k in keys:
        m = rex.search(k)
        if m:
            counts[k[:m.start()]] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _anima_depth_from_keys(keys) -> int:
    keys = list(keys)
    rex = _dit_block_re(keys)
    ns = anima_block_namespace(keys)
    hi = -1
    for k in keys:
        m = rex.search(k)
        if m and k[:m.start()] == ns:
            v = int(m.group(1))
            if v > hi:
                hi = v
    return (hi + 1) if hi >= 0 else ANIMA_NUM_BLOCKS


def _meta_trim_bw(bw, mode, model_name):
    """Block weights as actually applied, for the metadata.

    The UI always builds ANIMA_MAX_BLOCKS sliders; only the first N are used.
    """
    try:
        if str(mode).startswith("Anima"):
            return list(bw)[:_anima_ui_depth(model_name)]
    except Exception:
        pass
    return list(bw)


def _anima_ui_depth(model_name) -> int:
    """Real DiT depth of a model selected in a UI dropdown."""
    try:
        if model_name:
            pa = find_model(model_name)
            if pa:
                return anima_block_count(pa)
    except Exception:
        pass
    return ANIMA_NUM_BLOCKS


def anima_block_labels(n: int = None):
    n = ANIMA_NUM_BLOCKS if not n or n <= 0 else int(n)
    return [f"BLK_{i:02d}" for i in range(n)]


def fit_block_weights(bw, n: int, fill: float = 1.0):
    """Pad or truncate a block-weight list to exactly n entries.

    Padding repeats the last value (or `fill` for an empty list).
    """
    bw = [float(v) if v is not None else fill for v in (bw or [])]
    if len(bw) >= n:
        return bw[:n]
    pad = bw[-1] if bw else fill
    return bw + [pad] * (n - len(bw))


# ── Depth remap between Anima checkpoints of different block counts ──────────
#
# An upscale keeps the base blocks but shifts their indices (base block 6 can
# be block 9 in a 40-block model), so merging by raw index pairs unrelated
# layers. These helpers build an index map between two depths.

_REMAP_SIG_N = 4096
_REMAP_CACHE = {}

# ── Known depth-upscale recipes ─────────────────────────────────────────────
#
# Keyed by (deep, shallow). None marks a block inserted by the upscale, with no
# counterpart in the shallower model.
#
# (40, 28): Anima 2.9B. Same table as MAPPING in Forge Neo's extras.py; each
# inserted block starts as a copy of the block before it.
ANIMA_KNOWN_DEPTH_MAPS = {
    # (52, 40): Anima 3.8B, a LLaMA-Pro style expansion of the 2.9B. Taken from
    # the 3.8B's insertion_positions metadata; used when that metadata is missing.
    (52, 40): {0: 0, 1: 1, 2: 2, 3: None, 4: 3, 5: 4, 6: 5, 7: None,
               8: 6, 9: 7, 10: 8, 11: None, 12: 9, 13: 10, 14: 11, 15: None,
               16: 12, 17: 13, 18: 14, 19: None, 20: 15, 21: 16, 22: 17, 23: None,
               24: 18, 25: 19, 26: 20, 27: None, 28: 21, 29: 22, 30: 23, 31: None,
               32: 24, 33: 25, 34: 26, 35: None, 36: 27, 37: 28, 38: 29, 39: None,
               40: 30, 41: 31, 42: 32, 43: None, 44: 33, 45: 34, 46: 35, 47: None,
               48: 36, 49: 37, 50: 38, 51: 39},

    (40, 28): {
        0: 0,  1: 1,  2: None, 3: 2,  4: 3,  5: None, 6: 4,  7: 5,  8: None,
        9: 6,  10: 7, 11: None, 12: 8, 13: 9, 14: None, 15: 10, 16: 11, 17: None,
        18: 12, 19: 13, 20: 14, 21: None, 22: 15, 23: 16, 24: None,
        25: 17, 26: 18, 27: None, 28: 19, 29: 20, 30: None,
        31: 21, 32: 22, 33: None, 34: 23, 35: 24, 36: None,
        37: 25, 38: 26, 39: 27,
    },
}

# Map used before the adapter-namespace fix (differs in blocks 2-8). Selectable
# as 'legacy' to reproduce older merges.
ANIMA_LEGACY_DEPTH_MAP = {
    (40, 28): {
        0: 0,  1: 1,  2: 2,  3: 3,  4: 4,  5: 5,
        6: None, 7: None, 8: None,
        9: 6,  10: 7, 11: None, 12: 8, 13: 9, 14: None, 15: 10, 16: 11, 17: None,
        18: 12, 19: 13, 20: 14, 21: None, 22: 15, 23: 16, 24: None,
        25: 17, 26: 18, 27: None, 28: 19, 29: 20, 30: None,
        31: 21, 32: 22, 33: None, 34: 23, 35: 24, 36: None,
        37: 25, 38: 26, 39: 27,
    },
}


def _block_signature(sd, idx, prefix_re, want=6, ns=None):
    """Fingerprint of one block: a few sampled tensors from namespace `ns`."""
    import torch
    sigs = {}
    taken = 0
    for k in sd.keys():
        m = prefix_re.search(k)
        if not m or int(m.group(1)) != idx:
            continue
        if ns is not None and k[:m.start()] != ns:
            continue
        tail = k[m.end():]
        try:
            t = sd[k]
        except Exception:
            continue
        if not hasattr(t, "flatten"):
            continue
        v = t.flatten().float()
        n = v.numel()
        if n > _REMAP_SIG_N:
            step = max(1, n // _REMAP_SIG_N)
            v = v[::step][:_REMAP_SIG_N]
        sigs[tail] = v
        taken += 1
        if taken >= want:
            break
    return sigs


def _sig_cos(sa, sb):
    import torch
    common = [t for t in sa if t in sb and sa[t].numel() == sb[t].numel()]
    if not common:
        return -2.0
    tot, n = 0.0, 0
    for t in common:
        a, b = sa[t], sb[t]
        na = float(torch.linalg.norm(a))
        nb = float(torch.linalg.norm(b))
        if na == 0 or nb == 0:
            continue
        tot += float(torch.dot(a, b)) / (na * nb)
        n += 1
    return (tot / n) if n else -2.0


def anima_map_from_metadata(sd_deep, n_deep: int, n_shallow: int):
    """Depth map from the deeper checkpoint's own metadata.

    Anima expansions record where the new blocks were inserted:

        insertion_positions : [3, 7, 11, ...]   indices in the new stack
        old_block_count     : 40
        new_block_count     : 52

    Returns None when the metadata is missing or inconsistent.
    """
    try:
        meta = sd_deep.metadata() if hasattr(sd_deep, "metadata") else {}
    except Exception:
        meta = {}
    if not meta:
        return None

    raw = meta.get("insertion_positions")
    if not raw:
        return None
    try:
        positions = [int(x) for x in json.loads(raw)]
    except Exception:
        # Also accept a bare "3, 7, 11" style list.
        try:
            positions = [int(x) for x in str(raw).strip("[]() ").split(",") if x.strip()]
        except Exception:
            return None
    if not positions:
        return None

    # Cross-check against the recorded counts before trusting any of it.
    def _as_int(name):
        try:
            return int(meta.get(name))
        except (TypeError, ValueError):
            return None
    m_new, m_old = _as_int("new_block_count"), _as_int("old_block_count")
    if m_new is not None and m_new != n_deep:
        _log(f"⚠  Metadata says new_block_count={m_new} but the file has {n_deep} "
             f"DiT blocks — ignoring the recorded insertion positions.")
        return None
    if m_old is not None and m_old != n_shallow:
        _log(f"⚠  Metadata insertion positions describe an expansion from "
             f"{m_old} blocks, but Model B has {n_shallow} — ignoring them.")
        return None
    if len(positions) != n_deep - n_shallow:
        _log(f"⚠  Metadata lists {len(positions)} insertion positions but the depth "
             f"difference is {n_deep - n_shallow} — ignoring them.")
        return None

    ins = set(positions)
    mapping, seen = {}, 0
    for i in range(n_deep):
        if i in ins:
            mapping[i] = None
            seen += 1
        else:
            mapping[i] = i - seen
    mapped = [v for v in mapping.values() if v is not None]
    if sorted(mapped) != list(range(n_shallow)):
        _log("⚠  Metadata insertion positions do not produce a one-to-one map — "
             "ignoring them.")
        return None
    return mapping


def compose_depth_maps(outer: dict, inner: dict) -> dict:
    """Chain two depth maps: outer is deep->mid, inner is mid->shallow.

    Anima variants are successive expansions (28 -> 40 -> 52), so only
    consecutive pairs have a direct map. A block survives only if it is mapped
    in both hops.
    """
    return {i: (None if j is None else inner.get(j))
            for i, j in outer.items()}


def _direct_depth_map(sd_deep, n_deep: int, n_shallow: int, source: str):
    """The map for one pair, from metadata first, then the recipe table."""
    if source in ("auto", "known"):
        m = anima_map_from_metadata(sd_deep, n_deep, n_shallow)
        if m:
            return m, "metadata"
    known = ANIMA_KNOWN_DEPTH_MAPS.get((n_deep, n_shallow)) if source in ("auto", "known") else None
    if known:
        return dict(known), "recipe"
    return None, None


def resolve_depth_map(sd_deep, n_deep: int, n_shallow: int, source: str):
    """Map for (n_deep, n_shallow), chaining through an intermediate depth when
    there is no direct one (e.g. 52 -> 40 -> 28). Returns (map, how) or (None, None).
    """
    direct, how = _direct_depth_map(sd_deep, n_deep, n_shallow, source)
    if direct:
        return direct, how

    # Candidate intermediates: every depth the recipe table knows about.
    mids = sorted({d for pair in ANIMA_KNOWN_DEPTH_MAPS for d in pair},
                  reverse=True)
    for n_mid in mids:
        if not (n_shallow < n_mid < n_deep):
            continue
        outer, how_o = _direct_depth_map(sd_deep, n_deep, n_mid, source)
        inner = ANIMA_KNOWN_DEPTH_MAPS.get((n_mid, n_shallow))
        if outer and inner:
            composed = compose_depth_maps(outer, inner)
            mapped = [v for v in composed.values() if v is not None]
            if sorted(mapped) == list(range(n_shallow)):
                return composed, f"{how_o} {n_deep}->{n_mid} chained with recipe {n_mid}->{n_shallow}"
    return None, None


# How the last depth map was obtained, for logging.
_LAST_DEPTH_MAP_INFO = [""]

_VERIFY_PROBES = 6


def _verify_depth_map(sd_deep, sd_shallow, mapping, rex, ns_deep, ns_shallow):
    """Check a deep->shallow map against the weights of both models.

    For a few sampled pairs (i, j), deep block i must be closer to shallow
    block j than to j-1 and j+1. The test is relative because heavily trained
    expansions drift too far for a fixed cosine threshold.
    Returns (ok, details).
    """
    pairs = [(i, j) for i, j in sorted(mapping.items()) if j is not None]
    if not pairs:
        return False, "empty map"
    step = max(1, len(pairs) // _VERIFY_PROBES)
    probe = pairs[::step][:_VERIFY_PROBES]
    n_sh = max(j for _, j in pairs) + 1
    for i, j in probe:
        sd_i = _block_signature(sd_deep, i, rex, ns=ns_deep)
        c = _sig_cos(sd_i, _block_signature(sd_shallow, j, rex, ns=ns_shallow))
        for nb in (j - 1, j + 1):
            if 0 <= nb < n_sh:
                c_nb = _sig_cos(sd_i, _block_signature(sd_shallow, nb, rex, ns=ns_shallow))
                if c_nb > c:
                    return False, f"deep {i} matches shallow {nb} better than {j}"
    return True, f"{len(probe)} pairs checked"


def build_anima_depth_map(sd_a, sd_b, n_a=None, n_b=None, thresh=0.30,
                          source="auto"):
    """Map each block index of A onto the matching block of B.

    Returns {a_idx: b_idx or None}; None means the block has no counterpart in
    B (inserted by an upscale). Returns None when the depths are equal.
    """
    import re as _re
    import torch
    rex = _re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    n_a = n_a or _anima_depth_from_keys(list(sd_a.keys()))
    n_b = n_b or _anima_depth_from_keys(list(sd_b.keys()))
    if n_a == n_b:
        return None

    ns_a = anima_block_namespace(list(sd_a.keys()))
    ns_b = anima_block_namespace(list(sd_b.keys()))

    if source == "legacy":
        legacy = ANIMA_LEGACY_DEPTH_MAP.get((n_a, n_b))
        if legacy:
            return dict(legacy)

    # Preferred: the deeper model's metadata, or a known recipe that passes
    # verification against the weights.
    if source in ("auto", "known"):
        deep    = sd_a if n_a > n_b else sd_b
        shallow = sd_b if n_a > n_b else sd_a
        ns_deep    = ns_a if n_a > n_b else ns_b
        ns_shallow = ns_b if n_a > n_b else ns_a
        n_deep, n_shallow = max(n_a, n_b), min(n_a, n_b)
        resolved, how = resolve_depth_map(deep, n_deep, n_shallow, source)
        if resolved and how != "metadata":
            # A recipe is an assumption about how the model was built: verify it.
            ok, why = _verify_depth_map(deep, shallow, resolved, rex,
                                        ns_deep, ns_shallow)
            if ok:
                how = f"{how} — verified against the weights ({why})"
            else:
                _log(f"⚠  Block map from {how} does NOT fit these weights ({why}) — "
                     f"deriving the map from the weights instead.")
                resolved = None
        if resolved:
            _LAST_DEPTH_MAP_INFO[0] = how
            _log(f"   Block map: {how} ({n_deep}->{n_shallow}).")
            if n_a > n_b:
                return resolved
            # A is the SHALLOW one: invert, so A-index -> B-index.
            return {i: j for j, i in resolved.items() if i is not None}

    _LAST_DEPTH_MAP_INFO[0] = "derived from weight similarity"
    sigs_a = {i: _block_signature(sd_a, i, rex, ns=ns_a) for i in range(n_a)}
    sigs_b = {j: _block_signature(sd_b, j, rex, ns=ns_b) for j in range(n_b)}

    mapping = {}
    for i in range(n_a):
        best_j, best_c = None, -2.0
        for j in range(n_b):
            c = _sig_cos(sigs_a[i], sigs_b[j])
            if c > best_c:
                best_c, best_j = c, j
        mapping[i] = best_j if best_c >= thresh else None

    # Each B block can be claimed once: keep the best match and unmap the others
    # (usually an inserted copy of its neighbour).
    claimed = {}
    for i, j in mapping.items():
        if j is None:
            continue
        c = _sig_cos(sigs_a[i], sigs_b[j])
        if j not in claimed or c > claimed[j][1]:
            if j in claimed:
                mapping[claimed[j][0]] = None
            claimed[j] = (i, c)
        else:
            mapping[i] = None
    return mapping

# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK PROBE — in-memory single-block swap diagnostics
#  Replaces one block at a time with Model B weights (in RAM, no disk merge),
#  generates an image per block. Reuses the live txt2img settings.
# ═════════════════════════════════════════════════════════════════════════════

# Label -> substring to match in the LOADED diffusion model's keys.
PROBE_SDXL_BLOCKS = {
    **{f"IN{i:02d}": f"input_blocks.{i}." for i in range(9)},
    "MID": "middle_block.",
    **{f"OUT{i:02d}": f"output_blocks.{i}." for i in range(9)},
}
# Anima: DiT blocks only (TE + VAE are skipped automatically since their keys
# won't match 'blocks.N.'). Verify real layer names with the Inspect tab.
PROBE_ANIMA_BLOCKS = {f"BLK_{i:02d}": f"blocks.{i}." for i in range(ANIMA_NUM_BLOCKS)}

# Possible diffusion-model prefixes inside a checkpoint on disk.
PROBE_UNET_PREFIXES = ["", "model.diffusion_model.", "diffusion_model.", "net."]


def probe_block_map(arch: str, n_blocks: int = None) -> dict:
    """Label -> key-substring map for the Block Probe, sized to n_blocks for Anima."""
    if is_dit_arch(arch):
        stem = "layers" if str(arch).lower() == "moss" else "blocks"
        if stem == "blocks" and (not n_blocks or int(n_blocks) == ANIMA_NUM_BLOCKS):
            return PROBE_ANIMA_BLOCKS
        return {f"BLK_{i:02d}": f"{stem}.{i}." for i in range(int(n_blocks or ANIMA_NUM_BLOCKS))}
    return PROBE_SDXL_BLOCKS


def probe_get_unet_module(model):
    """PyTorch module of the active diffusion model (the one with .state_dict())."""
    candidates = [
        lambda m: m.forge_objects.unet.model.diffusion_model,
        lambda m: m.forge_objects.unet.model,
        lambda m: m.model.diffusion_model,
        lambda m: m,
    ]
    for get in candidates:
        try:
            mod = get(model)
            if hasattr(mod, "state_dict") and callable(mod.state_dict):
                if len(mod.state_dict()) > 0:
                    return mod
        except Exception:
            continue
    raise RuntimeError("Diffusion model not found for probe.")


def probe_resolve(key: str, source_state: dict):
    """Find B's tensor matching a loaded-model key, trying known prefixes."""
    for pre in PROBE_UNET_PREFIXES:
        cand = pre + key
        if cand in source_state:
            return source_state[cand]
    return None


def probe_keys_for(unet_sd, prefix):
    # Whole path segments only: "blocks.1." must not hit Krea 2's layerwise_blocks.1.
    dotted = "." + prefix
    return [k for k in unet_sd.keys() if k.startswith(prefix) or dotted in k]


def probe_apply(unet_sd, prefix, source_state, alpha=1.0, a_backup=None):
    """Write B's weights into the block (in-place). Returns n. tensors applied."""
    applied = 0
    for k in probe_keys_for(unet_sd, prefix):
        src = probe_resolve(k, source_state)
        if src is None:
            continue
        src = src.to(unet_sd[k].dtype).to(unet_sd[k].device)
        if a_backup is not None and alpha < 1.0:
            unet_sd[k].copy_((1.0 - alpha) * a_backup[k] + alpha * src)
        else:
            unet_sd[k].copy_(src)
        applied += 1
    return applied


def probe_backup(unet_sd, prefix):
    return {k: unet_sd[k].detach().clone() for k in probe_keys_for(unet_sd, prefix)}


def probe_restore(unet_sd, backup):
    for k, v in backup.items():
        unet_sd[k].copy_(v)


# ── Group helpers: apply SEVERAL blocks at once into a single image ───────────

# Translate EASY_CATEGORIES block names (input_block_4, middle_block, base...)
# into probe key-prefixes (input_blocks.4., middle_block., time_embed./label_emb.).
def _easy_name_to_prefixes(block_name: str):
    if block_name == "middle_block":
        return ["middle_block."]
    if block_name == "base":
        # "base" = the non-block stem (time/label embeddings). Best-effort.
        return ["time_embed.", "label_emb."]
    if block_name.startswith("input_block_"):
        return [f"input_blocks.{block_name.rsplit('_', 1)[1]}."]
    if block_name.startswith("output_block_"):
        return [f"output_blocks.{block_name.rsplit('_', 1)[1]}."]
    return []


def easy_group_prefixes(category_name: str):
    """All probe key-prefixes covered by one semantic Easy category."""
    cat = EASY_CATEGORIES.get(category_name)
    if not cat:
        return []
    prefixes = []
    for bname in cat["blocks"].keys():
        prefixes.extend(_easy_name_to_prefixes(bname))
    # de-dup, keep order
    seen, out = set(), []
    for p in prefixes:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def probe_backup_many(unet_sd, prefixes):
    bk = {}
    for prefix in prefixes:
        for k in probe_keys_for(unet_sd, prefix):
            if k not in bk:
                bk[k] = unet_sd[k].detach().clone()
    return bk


def probe_apply_many(unet_sd, prefixes, source_state, alpha=1.0, a_backup=None):
    """Apply B to several blocks at once. Returns total tensors applied."""
    total = 0
    for prefix in prefixes:
        total += probe_apply(unet_sd, prefix, source_state, alpha=alpha, a_backup=a_backup)
    return total


def run_block_probe(model_b, arch, mode, blocks, easy_groups, alpha,
                    t2i_prompt, t2i_neg, t2i_steps, t2i_cfg,
                    t2i_w, t2i_h, t2i_seed, t2i_sampler):
    """Probe in three modes:
       - 'single' : one block per image (default)
       - 'easy'   : one image per semantic group (Style/Anatomy/...)
       - 'all'    : all selected blocks together in a single image
       In every mode: apply B -> generate -> restore. No disk merge, no reload.
    """
    import random
    from contextlib import closing
    import modules.scripts as _scripts_mod

    if not model_b:
        return "❌  Select Model B.", None

    pb = find_model(model_b)
    if not os.path.isfile(pb):
        return f"❌  File not found: {pb}", None

    # Size the block map to the model, like the UI does.
    _probe_depth = anima_block_count(pb) if is_dit_arch(arch) else None

    prompt  = (t2i_prompt or "").strip() or "masterpiece, best quality, 1girl, portrait"
    neg     = (t2i_neg or "").strip()
    sampler = (t2i_sampler or "Euler a").strip()
    steps   = max(1, int(float(t2i_steps or 28)))
    cfg     = float(t2i_cfg or 7.0)
    width   = int(float(t2i_w or 1024))
    height  = int(float(t2i_h or 1024))
    try:
        seed = int(float(t2i_seed))
    except (TypeError, ValueError):
        seed = -1
    if seed == -1:
        seed = random.randint(0, 2**31 - 1)   # fix the seed: comparable cells

    try:
        # Lazy open: only the keys of the selected blocks are read.
        b_state = open_model(pb, lazy=True)
        unet = probe_get_unet_module(shared.sd_model)
        unet_sd = unet.state_dict()
    except Exception as e:
        return f"❌  Probe setup failed: {e}", None

    # Build the list of "cells": each cell = (caption, [prefixes to apply]).
    cells = []
    mode = (mode or "single").lower()

    if mode.startswith("easy"):
        groups = easy_groups or [c for c in EASY_CATEGORIES if not c.startswith("⚗️")]
        for g in groups:
            prefixes = easy_group_prefixes(g)
            if prefixes:
                # short caption from the category name (before any slash)
                short = g.split(" /")[0].split(" [")[0]
                cells.append((short, prefixes))
    elif mode.startswith("all"):
        bmap = probe_block_map(arch, _probe_depth)
        sel = [b for b in (blocks or []) if b in bmap]
        if not sel:
            return "❌  Select some blocks for 'All together' mode.", None
        prefixes = [bmap[b] for b in sel]
        cells.append(("+".join(sel), prefixes))
    else:  # single
        bmap = probe_block_map(arch, _probe_depth)
        sub = {k: v for k, v in bmap.items() if (not blocks or k in set(blocks))}
        if not sub:
            return "❌  No valid block selected.", None
        for name, prefix in sub.items():
            cells.append((name, [prefix]))

    _log(f"Block Probe — B={os.path.basename(pb)} arch={arch} mode={mode} "
         f"seed={seed} cells={[c[0] for c in cells]}")

    def _make_p():
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model        = shared.sd_model,
            outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
            outpath_grids   = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
            prompt          = prompt, negative_prompt = neg,
            sampler_name    = sampler, steps = steps, cfg_scale = cfg,
            width = width, height = height, seed = seed,
            n_iter = 1, batch_size = 1,
        )
        try:
            p.scripts = _scripts_mod.ScriptRunner()
            p.script_args = ()
        except Exception:
            p.scripts = None
            p.script_args = ()
        return p

    results, captions = [], []

    # reference (Model A untouched)
    _log("Block Probe — reference...")
    with closing(_make_p()) as p:
        proc = processing.process_images(p)
        if proc and proc.images:
            results.append(proc.images[0]); captions.append("A (ref)")

    # one cell at a time (a cell may cover several blocks)
    for caption, prefixes in cells:
        backup = probe_backup_many(unet_sd, prefixes)
        try:
            n = probe_apply_many(unet_sd, prefixes, b_state, alpha=alpha, a_backup=backup)
            _log(f"Block Probe — {caption}: {n} tensors @ alpha={alpha}")
            if n == 0:
                continue
            with closing(_make_p()) as p:
                proc = processing.process_images(p)
                if proc and proc.images:
                    results.append(proc.images[0]); captions.append(f"{caption}@{alpha}")
        finally:
            probe_restore(unet_sd, backup)   # always restore A

    _log("Block Probe — done.")
    close_model(b_state)
    gallery = [(img, cap) for img, cap in zip(results, captions)]
    return (f"✅  Probe done — {len(gallery)} images · seed {seed}", gallery)

def detect_arch(path: str) -> str:
    """Quick architecture detection by sampling keys."""
    try:
        if path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                sample = list(sf.keys())
        else:
            sd = load_sd(path)
            sample = list(sd.keys())
        # Before Anima: Comfy-packaged Krea 2 also uses model.diffusion_model.blocks.
        if any(k.endswith("txtfusion.projector.weight") for k in sample):
            return "krea2"
        if _is_moss_keys(sample):
            return "moss"
        # Scan all keys: the VAE is often serialised first and would hide the
        # DiT keys from a small sample. Anima blocks can live under net.blocks.,
        # model.diffusion_model.blocks. or diffusion_model.blocks.
        if any(("net.blocks." in k or
                "diffusion_model.blocks." in k or
                k.startswith("model.diffusion_model.blocks.")) and "blocks." in k
               for k in sample):
            # Disambiguate: Anima DiT blocks carry adaln_modulation / q_proj naming.
            if any("adaln_modulation" in k or ".cross_attn.q_proj" in k for k in sample):
                return "anima"
        if any(k.startswith("net.blocks.") for k in sample):
            return "anima"
        if any("conditioner" in k or "input_blocks" in k for k in sample):
            return "sdxl"
    except Exception:
        pass
    return "sdxl"

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

def get_anima_block_index(key: str) -> int:
    """Block index of an Anima key, for any of the three save prefixes
    (net., diffusion_model., model.diffusion_model.)."""
    for prefix in ("net.blocks.", "model.diffusion_model.blocks.", "diffusion_model.blocks."):
        if prefix in key:
            try:
                return int(key.split(prefix, 1)[1].split(".")[0])
            except (ValueError, IndexError):
                pass
    return 0


def get_dit_block_index(key: str, arch: str) -> int:
    """Main-stack block index of a DiT key; 0 for everything else, as for Anima."""
    if arch == "anima":
        return get_anima_block_index(key)
    m = _dit_block_re(arch=arch).search(key)
    if m and key[:m.start()] in _DIT_TOP_PREFIXES:
        return int(m.group(1))
    return 0


def block_index_fn_for(arch: str):
    if arch in DIT_ARCHS:
        return lambda key: get_dit_block_index(key, arch)
    return get_ckpt_block_index

# ── LoRA blocks (SDXL naming)────────────────────────────────────────────────
# SDXL UNet LoRA key patterns → logical block index (same 20-slot schema)
LORA_BLOCK_PATTERNS = [
    # (pattern_substring, block_index)
    # BASE: text encoder + misc
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
    # Anima, dotted naming (most specific prefix first).
    for prefix in ("net.blocks.", "model.diffusion_model.blocks.", "diffusion_model.blocks."):
        if prefix in key:
            try:
                return int(key.split(prefix, 1)[1].split(".")[0])
            except (ValueError, IndexError):
                pass
    # Anima, kohya naming: lora_unet_blocks_N_...
    if key.startswith("lora_unet_blocks_"):
        try:
            return int(key[len("lora_unet_blocks_"):].split("_", 1)[0])
        except (ValueError, IndexError):
            pass
    key_lower = key.lower()
    for pattern, idx in LORA_BLOCK_PATTERNS:
        if pattern in key_lower:
            return idx
    return 0  # fallback: BASE

# ═════════════════════════════════════════════════════════════════════════════
#  SEMANTIC CATEGORIES  (shared between Block Merge and LoRA Tools)
# ═════════════════════════════════════════════════════════════════════════════

# ── Anima (DiT) semantic categories ─────────────────────────────────────────
# From Crody's / Team-C per-layer breakdown of Anima's 28 DiT blocks (Model
# Merge Guide v3.0). The blocks are sequential, with roles mapped empirically:
#   L00-L03  input-side detail, fingers, outlines, local stability
#   L04-L08  body shape, clothing, base structure, local form
#   L09-L13  face base, body shape, pose support, composition bridge
#   L14-L17  eyes, facial impression, age impression, expression core
#   L18-L21  nose, mouth, jaw, lower face, expression, paint support
#   L22-L24  style reflection, texture, background density, paint support
#   L25-L27  final look, paint, saturation, contrast, default style
# The weights are each block's emphasis within a category; the slider scales them.
ANIMA_EASY_CATEGORIES = {
    "Hands / Line quality": {
        "icon": "✋",
        "description": "Fingers, outlines, local stability, noise reduction",
        "blocks": {0: 0.9, 1: 1.0, 2: 0.5, 3: 0.3, 8: 0.4},
    },
    "Body / Clothing": {
        "icon": "🦴",
        "description": "Physique, limbs, clothing structure, hair cohesion",
        "blocks": {3: 0.4, 4: 1.0, 5: 0.9, 6: 0.8, 7: 0.6, 8: 0.7},
    },
    "Pose / Composition": {
        "icon": "🕺",
        "description": "Posture, limb attachment, overall framing",
        "blocks": {6: 0.7, 7: 0.9, 8: 0.8, 9: 1.0, 13: 0.5},
    },
    "Face structure": {
        "icon": "🙂",
        "description": "Face outline, cheeks, jaw width, age impression",
        "blocks": {9: 0.5, 10: 1.0, 11: 0.9, 12: 0.8, 13: 0.7},
    },
    "Eyes / Expression": {
        "icon": "👁️",
        "description": "Eye shape, pupils, facial impression, expression core",
        "blocks": {14: 1.0, 15: 0.9, 16: 0.8, 17: 0.9},
    },
    "Mouth / Lower face": {
        "icon": "👄",
        "description": "Nose, mouth, jaw, lower-face expression",
        "blocks": {18: 1.0, 19: 0.9, 20: 0.7, 21: 0.6},
    },
    "Background / Detail density": {
        "icon": "🌄",
        "description": "Background depth, small-object density, drawing detail",
        "blocks": {2: 0.8, 22: 0.9, 23: 1.0},
    },
    "Style / Texture": {
        "icon": "🖌️",
        "description": "Style reflection, paint, shading, line sharpness",
        "blocks": {21: 0.4, 22: 0.8, 23: 0.7, 24: 1.0, 25: 0.9},
    },
    "Color / Final look": {
        "icon": "🎨",
        "description": "Saturation, contrast, gloss, default-style lock",
        "blocks": {25: 0.8, 26: 1.0, 27: 1.0},
    },
}


def anima_easy_to_block_weights(slider_values: dict, n_blocks: int = None) -> list:
    """Semantic sliders -> per-block weights for Anima.

    Overlapping categories are averaged, not summed, so two sliders at 0.5
    don't push a shared block to 1.0.
    """
    n = n_blocks or ANIMA_NUM_BLOCKS
    if n != ANIMA_NUM_BLOCKS:
        return _anima_easy_deep(slider_values, n)
    contributions = [0.0] * n
    counts        = [0]   * n
    for cat_name, val in slider_values.items():
        info = ANIMA_EASY_CATEGORIES.get(cat_name)
        if not info:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        for idx, base_w in info["blocks"].items():
            if 0 <= idx < n:
                contributions[idx] += base_w * val
                counts[idx]        += 1
    bw = [
        min(1.0, contributions[i] / counts[i]) if counts[i] > 0 else 0.0
        for i in range(n)
    ]
    return bw


def _anima_easy_deep(slider_values: dict, n: int) -> list:
    """Semantic sliders for a depth-upscaled Anima (40 or 52 blocks).

    The table describes the 28 base blocks, so it is translated through the
    upscale recipe. Inserted blocks take the weight of the block before them,
    which they start as a copy of. Unknown depths use a proportional spread.
    """
    base = anima_easy_to_block_weights(slider_values, ANIMA_NUM_BLOCKS)
    m, _how = resolve_depth_map(None, n, ANIMA_NUM_BLOCKS, "known") \
        if n > ANIMA_NUM_BLOCKS else (None, None)
    if m:
        out, last = [], base[0]
        for i in range(n):
            j = m.get(i)
            if j is not None:
                last = base[j]
            out.append(last)
        return out
    return [base[round(i * (ANIMA_NUM_BLOCKS - 1) / max(n - 1, 1))]
            for i in range(n)]


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
    # Heuristic mappings from community observations; results vary between models.

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

def easy_to_block_weights(slider_values: dict, use_experimental: bool = False) -> list:
    """Semantic sliders → 20 per-block weights.

    When use_experimental is False (default), experimental sliders are skipped
    entirely so they don't dilute the average of the main categories.
    """
    contributions = [0.0] * 20
    counts        = [0]   * 20
    for cat_name, val in slider_values.items():
        if cat_name not in EASY_CATEGORIES:
            continue
        if not use_experimental and cat_name.startswith("⚗️ [EXP]"):
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

_SF_DTYPE_TAGS = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}
if hasattr(torch, "float8_e4m3fn"):
    _SF_DTYPE_TAGS[torch.float8_e4m3fn] = "F8_E4M3"
if hasattr(torch, "float8_e5m2"):
    _SF_DTYPE_TAGS[torch.float8_e5m2] = "F8_E5M2"
_SF_TAG_DTYPES = {v: k for k, v in _SF_DTYPE_TAGS.items()}

# Scaled fp8 (Comfy weight_scale / legacy scale_weight): raw values of two files
# sit on different scales, so they are dequantized on read and the scale keys
# hidden. Other quantized formats are refused by the merge engine.
_FP8_QUANT_FORMATS = {"float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2"}
_QUANT_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
_QUANT_AUX_SUFFIXES = (".input_scale", ".scale_input", ".comfy_quant")


class _LazySafetensors:
    """Read-only, lazy view over a .safetensors file.

    Tensors are read from disk one key at a time. Supports the dict operations
    the engine uses (keys, in, []). Other formats go through load_sd().
    Scaled fp8 weights come back dequantized to float32.
    """
    def __init__(self, path: str):
        from safetensors import safe_open
        self._path = path
        self._f = safe_open(path, framework="pt", device="cpu")
        self._scale = {}
        self._fp8_view = {}
        self.quant_formats = set()
        all_keys = list(self._f.keys())
        hidden = self._scan_quant(all_keys)
        self._order = [k for k in all_keys if k not in hidden] if hidden else all_keys
        self._keys = set(self._order)

    def _tag(self, key):
        return str(self._f.get_slice(key).get_dtype()).upper()

    def _scan_quant(self, all_keys):
        fmt_by_layer = {}
        for k in all_keys:
            if k.endswith(".comfy_quant"):
                try:
                    conf = json.loads(bytes(self._f.get_tensor(k).numpy().tobytes()).decode("utf-8"))
                    fmt_by_layer[k[:-len(".comfy_quant")]] = conf.get("format", "unknown")
                except Exception:
                    fmt_by_layer[k[:-len(".comfy_quant")]] = "unknown"
        try:
            qmeta = json.loads((self.metadata() or {}).get("_quantization_metadata") or "{}") or {}
        except Exception:
            qmeta = {}
        layers = qmeta.get("layers") or {}
        if layers:
            first = next(iter(layers))
            prefix = next((k[:-len(first + ".weight")] for k in all_keys
                           if k.endswith(first + ".weight")), "")
            for layer, conf in layers.items():
                fmt_by_layer.setdefault(prefix + layer,
                                        (conf or {}).get("format", "unknown"))
        if any(k.endswith("scaled_fp8") for k in all_keys):
            for k in all_keys:
                if k.endswith(".scale_weight"):
                    fmt_by_layer.setdefault(k[:-len(".scale_weight")], "float8_e4m3fn")
        if any(k.endswith(".qweight") for k in all_keys):
            self.quant_formats.add("nunchaku")
        if any(k.endswith(".absmax") or ".quant_state." in k for k in all_keys):
            self.quant_formats.add("bitsandbytes")

        key_set = set(all_keys)
        hidden = {k for k in all_keys if k.endswith("scaled_fp8")}
        fp8_layers = set()
        for k in all_keys:
            for suf in _QUANT_SCALE_SUFFIXES:
                if not k.endswith(suf):
                    continue
                layer = k[:-len(suf)]
                wk = layer + ".weight"
                if wk not in key_set:
                    continue
                fmt = fmt_by_layer.get(layer)
                if fmt is None and self._tag(wk).startswith("F8"):
                    fmt = "float8_e4m3fn"
                if fmt in _FP8_QUANT_FORMATS:
                    self._scale[wk] = k
                    fp8_layers.add(layer)
        for layer, fmt in fmt_by_layer.items():
            self.quant_formats.add(fmt)
            if fmt not in _FP8_QUANT_FORMATS:
                continue
            fp8_layers.add(layer)
            wk = layer + ".weight"
            if wk in key_set and self._tag(wk) == "U8":
                dt = getattr(torch, _FP8_QUANT_FORMATS[fmt], None)
                if dt is not None:
                    self._fp8_view[wk] = dt
        for layer in fp8_layers:
            for suf in _QUANT_SCALE_SUFFIXES + _QUANT_AUX_SUFFIXES:
                if layer + suf in key_set:
                    hidden.add(layer + suf)
        if self._scale or self._fp8_view:
            self.quant_formats.add("float8_e4m3fn")
        return hidden

    @property
    def unsupported_quant(self):
        return sorted(f for f in self.quant_formats if f not in _FP8_QUANT_FORMATS)

    @property
    def dequantized(self) -> int:
        return len(set(self._scale) | set(self._fp8_view))

    def keys(self):
        return self._order

    def metadata(self):
        """Header metadata dict, or {}."""
        try:
            return self._f.metadata() or {}
        except Exception:
            return {}

    def shape_of(self, key):
        """Tensor shape from the header, without reading the tensor."""
        return tuple(self._f.get_slice(key).get_shape())

    def dtype_of(self, key):
        """Dtype of what [] returns, from the header."""
        if key in self._scale or key in self._fp8_view:
            return torch.float32
        return _SF_TAG_DTYPES.get(self._tag(key))

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        t = self._f.get_tensor(key)
        view = self._fp8_view.get(key)
        if view is not None:
            t = t.view(view)
        skey = self._scale.get(key)
        if skey is not None:
            s = self._f.get_tensor(skey).float()
            t = t.float()
            if s.numel() == 1:
                t = t * s.reshape(())
            elif t.dim() > 0 and s.numel() == t.shape[0]:
                t = t * s.reshape(-1, *([1] * (t.dim() - 1)))
            else:
                t = t * s
        elif view is not None:
            t = t.float()
        return t

    def close(self):
        # safe_open handle is released when the object is dropped.
        self._f = None
        self._keys = set()
        self._order = []


def open_model(path: str, lazy: bool = True):
    """Mapping-like view of a model's tensors: lazy for .safetensors, else load_sd()."""
    if lazy and path.endswith(".safetensors"):
        try:
            return _LazySafetensors(path)
        except Exception as e:
            _log(f"Lazy open failed ({e}); falling back to full load.")
            return load_sd(path)
    return load_sd(path)


def close_model(obj):
    """Release a model opened with open_model()."""
    try:
        if isinstance(obj, _LazySafetensors):
            obj.close()
    except Exception:
        pass


# Top-level prefixes used by different Anima save formats. Models saved with
# different prefixes are matched on the key tail instead of the full name.
_KEY_NORMALISE_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "net.",
)

def _normalise_key(key: str) -> str:
    """Strip a known top-level prefix so keys from different save formats align."""
    for p in _KEY_NORMALISE_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key

def _build_normalised_lookup(model_keys):
    """Map normalised tail -> real key for one model (first key wins on collisions)."""
    out = {}
    for k in model_keys:
        nk = _normalise_key(k)
        if nk not in out:
            out[nk] = k
    return out

def cast_precision(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Cast a tensor to the requested precision.

    Non-floating tensors (position_ids, masks, ...) are returned unchanged.
    """
    if not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=precision_dtype(precision))


def precision_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8":  torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    }
    return mapping.get(precision, torch.float16)

def save_sf(state_dict: dict, path: str, precision: str = "fp16", metadata: dict = None):
    """Save a state dict as safetensors, with precision cast and metadata.

    Casts in place, one tensor at a time, so two copies are never held in RAM.
    Mutates the caller's dict.
    """
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in list(state_dict.keys()):
        state_dict[k] = cast_precision(state_dict[k], precision).contiguous()
    meta = {k: str(v) for k, v in metadata.items()} if metadata else {}
    save_file(state_dict, path, metadata=meta)


# Write merge output tensor by tensor instead of holding it in RAM.
# False = in-memory save_sf path.
STREAMING_SAVE = True

VAE_PREFIX = "first_stage_model."


class StreamingSafetensorsWriter:
    """Writes a .safetensors file one tensor at a time.

    plan: list of (key, torch.dtype, shape). The header, with every offset, is
    written up front; write() is then called once per key, in plan order.
    Data goes to `<path>.part`, renamed over `path` only by finish(), so a
    stopped or failed merge never leaves a truncated model behind.
    """

    def __init__(self, path: str, plan: list, metadata: dict = None):
        import struct
        self.path = path
        self.tmp = path + ".part"
        self._order = []
        self._spec = {}
        header = {}
        if metadata:
            header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
        offset = 0
        for key, dtype, shape in plan:
            if dtype not in _SF_DTYPE_TAGS:
                raise ValueError(f"unsupported dtype {dtype} on '{key}'")
            if key in self._spec:
                raise ValueError(f"duplicate key '{key}' in the output plan")
            shape = tuple(int(s) for s in shape)
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            header[key] = {"dtype": _SF_DTYPE_TAGS[dtype], "shape": list(shape),
                           "data_offsets": [offset, offset + nbytes]}
            self._order.append(key)
            self._spec[key] = (dtype, shape)
            offset += nbytes
        self.data_bytes = offset
        hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hb += b" " * ((8 - len(hb) % 8) % 8)
        self._header_len = len(hb)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(self.tmp, "wb")
        self._f.write(struct.pack("<Q", len(hb)))
        self._f.write(hb)
        self._i = 0

    @property
    def total_bytes(self) -> int:
        return 8 + self._header_len + self.data_bytes

    def write(self, key: str, tensor: torch.Tensor):
        if self._i >= len(self._order) or self._order[self._i] != key:
            exp = self._order[self._i] if self._i < len(self._order) else "<end>"
            raise RuntimeError(f"Streaming save: got '{key}', expected '{exp}'")
        dtype, shape = self._spec[key]
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype != dtype:
            t = t.to(dtype)
        if tuple(t.shape) != shape:
            raise ValueError(f"Streaming save: '{key}' has shape {tuple(t.shape)}, "
                             f"planned {shape}")
        t = t.contiguous().reshape(-1)
        if t.numel():
            self._f.write(t.view(torch.uint8).numpy().data)
        self._i += 1

    def finish(self):
        """Close, check the size against the header and move into place."""
        import gc
        if self._i != len(self._order):
            raise RuntimeError(f"Streaming save: {self._i}/{len(self._order)} tensors written")
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        size = os.path.getsize(self.tmp)
        if size != self.total_bytes:
            raise RuntimeError(f"Streaming save: file is {size} bytes, header "
                               f"says {self.total_bytes}")
        gc.collect()        # drops a lingering mmap of the target on Windows
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            if not self._f.closed:
                self._f.close()
        finally:
            try:
                os.remove(self.tmp)
            except OSError:
                pass


def plan_streaming_output(sd_a, keys, precision: str, vae_items=None):
    """Output plan of a merge: A's keys at the target precision, VAE swapped in."""
    out_dt = precision_dtype(precision)
    plan = []
    for k in keys:
        if vae_items is not None and k.startswith(VAE_PREFIX):
            continue
        dt = sd_a.dtype_of(k)
        if dt is None:
            raise ValueError(f"unknown dtype on '{k}'")
        plan.append((k, out_dt if dt.is_floating_point else dt, sd_a.shape_of(k)))
    for k, v in (vae_items or []):
        plan.append((k, out_dt if v.is_floating_point() else v.dtype, tuple(v.shape)))
    return plan


def load_vae_items(vae_path: str) -> list:
    """(key, tensor) pairs of a VAE file, renamed under VAE_PREFIX."""
    vae_sd = load_sd(vae_path)
    return list({(k if k.startswith(VAE_PREFIX) else VAE_PREFIX + k): v
                 for k, v in vae_sd.items()}.items())

# SHA256 cache keyed on (path, size, mtime).
_HASH_CACHE = {}

def compute_hash(path: str) -> str:
    """Compute short SHA256 hash of a file (first 8 hex chars). Cached."""
    import hashlib
    st = os.stat(path)
    cache_key = (path, st.st_size, int(st.st_mtime))
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()[:8]
    _HASH_CACHE[cache_key] = digest
    return digest

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
    """Exact q-quantile of |tensor|.

    torch.kthvalue avoids the full sort torch.quantile does and is much faster
    on large tensors. Falls back to a random sample if kthvalue fails.
    """
    import time as _t
    _t0 = _t.time()
    flat = tensor.abs().flatten()
    n    = flat.numel()

    if n == 0:
        _QUANTILE_TIME[0] += _t.time() - _t0
        return torch.tensor(0.0, device=tensor.device)

    try:
        # k is 1-indexed: lower-interpolation quantile.
        k = int(q * (n - 1)) + 1
        k = max(1, min(n, k))
        out = torch.kthvalue(flat, k).values
        _QUANTILE_TIME[0] += _t.time() - _t0
        return out
    except Exception:
        # Fallback: random sample (very large or unsupported tensors).
        sample = min(n, 1_000_000)
        idx = torch.randint(0, n, (sample,), device=flat.device)
        out = flat[idx].quantile(q)
        _QUANTILE_TIME[0] += _t.time() - _t0
        return out

def ties_merge(tensors: list, alphas: list, density: float = 0.5) -> torch.Tensor:
    """
    TIES merge: Trim, Elect Sign, Merge.
    tensors: list of (base, other) pairs
    alphas:  per-tensor weights
    density: fraction of weights to keep (trim threshold)
    """
    base = tensors[0][0].float()
    task_vectors = [(t.float() - base) * a for (_, t), a in zip(tensors, alphas)]

    # Trim in place to avoid a second copy of every task vector.
    trimmed = task_vectors
    for tv in trimmed:
        threshold = _quantile_safe(tv, 1.0 - density)
        tv *= (tv.abs() >= threshold)

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

# ── Task-vector methods ──────────────────────────────────────────────────────
# Same contract as ties_merge / dare_merge: float base and donor tensors in,
# float tensor out (the caller reshapes it to A's shape).

def task_arithmetic_merge(base: torch.Tensor, others: list, alphas: list) -> torch.Tensor:
    """Task Arithmetic: base + Σ alpha_i · (other_i - base).

    The unfiltered task-vector sum. A negative alpha removes a direction (e.g.
    a style). On heavily merged checkpoints it also amplifies leftover noise;
    Breadcrumbs, DELLA or TIES suit those better.
    """
    base_f = base.float()
    result = base_f.clone()
    for other, alpha in zip(others, alphas):
        tv = other.float() - base_f
        result = result + alpha * tv
    return result


def breadcrumbs_merge(base: torch.Tensor, others: list, alphas: list,
                      density: float = 0.9, gamma: float = 0.01,
                      use_ties: bool = False) -> torch.Tensor:
    """Model Breadcrumbs: task arithmetic with a two-sided magnitude mask.

    Each task vector loses both its largest outliers (top `gamma`) and its
    smallest values, keeping the middle band, optionally followed by a TIES
    sign election. Meant for noisy, heavily merged checkpoints.

    density:  fraction of weights kept after both cuts
    gamma:    fraction cut from the high (outlier) tail
    use_ties: elect a dominant sign across vectors
    """
    base_f = base.float()

    masked_tvs = []
    for other, alpha in zip(others, alphas):
        tv = (other.float() - base_f) * alpha
        absflat = tv.abs()

        # High cut: remove the top `gamma` fraction (big outliers).
        hi_thr = _quantile_safe(absflat, 1.0 - gamma)
        # Low cut: remove the bottom part so that roughly `density` survives
        # between the two cuts. low_q is positioned below the high cut.
        low_q = max(0.0, 1.0 - gamma - density)
        lo_thr = _quantile_safe(absflat, low_q)

        keep = (tv.abs() >= lo_thr) & (tv.abs() <= hi_thr)
        masked_tvs.append(tv * keep)

    if not masked_tvs:
        return base_f.clone()

    if use_ties:
        sign_sum = sum(t.sign() for t in masked_tvs)
        elected  = sign_sum.sign()
        merged = torch.zeros_like(base_f)
        count  = torch.zeros_like(base_f)
        for t in masked_tvs:
            agree  = (t.sign() == elected) | (elected == 0)
            merged = merged + t * agree
            count  = count  + agree.float()
        count = count.clamp(min=1)
        return base_f + merged / count

    merged = sum(masked_tvs)
    return base_f + merged


def della_merge(base: torch.Tensor, others: list, alphas: list,
                density: float = 0.7, epsilon: float = 0.1,
                use_ties: bool = False) -> torch.Tensor:
    """DELLA: DARE with magnitude-ranked drop probabilities.

    Keep probabilities range from (density - epsilon) for the smallest changes
    to (density + epsilon) for the largest, so important changes survive more
    often. Survivors are rescaled as in DARE, optionally followed by a TIES
    sign election.

    density:  average keep probability
    epsilon:  spread of the keep probability across the ranking
    use_ties: elect a dominant sign across vectors
    """
    base_f = base.float()

    processed = []
    for other, alpha in zip(others, alphas):
        tv = other.float() - base_f
        flat = tv.flatten()
        n = flat.numel()

        # Rank by magnitude (0 = smallest .. 1 = largest).
        absflat = flat.abs()
        if n <= 262_144:
            # Exact ranks: argsort is cheap on small tensors.
            order = torch.argsort(absflat)          # ascending
            ranks = torch.empty_like(order, dtype=torch.float)
            ranks[order] = torch.arange(n, device=flat.device, dtype=torch.float)
            norm_rank = ranks / max(n - 1, 1)       # 0..1, large magnitude -> ~1
        else:
            # Large tensors: approximate ranks with 32 quantile buckets from a
            # random sample instead of a full argsort. The keep probability is
            # quantised to 32 levels (error < epsilon/16).
            B = 32
            idx = torch.randint(0, n, (262_144,), device=flat.device)
            qs = torch.linspace(1.0 / B, (B - 1) / B, B - 1,
                                device=flat.device, dtype=absflat.dtype)
            thresholds = torch.quantile(absflat[idx], qs)
            norm_rank = torch.bucketize(absflat, thresholds).float() / (B - 1)

        # Keep prob: smaller magnitude -> lower keep, larger -> higher keep.
        keep_prob = (density - epsilon) + (2.0 * epsilon) * norm_rank
        keep_prob = keep_prob.clamp(0.0, 1.0)

        mask = (torch.rand_like(flat) < keep_prob).float()
        # Rescale survivors by their own keep prob (DARE-style, per-element).
        kept = flat * mask / (keep_prob + 1e-8)
        processed.append((kept.reshape(tv.shape)) * alpha)

    if not processed:
        return base_f.clone()

    if use_ties:
        sign_sum = sum(t.sign() for t in processed)
        elected  = sign_sum.sign()
        merged = torch.zeros_like(base_f)
        count  = torch.zeros_like(base_f)
        for t in processed:
            agree  = (t.sign() == elected) | (elected == 0)
            merged = merged + t * agree
            count  = count  + agree.float()
        count = count.clamp(min=1)
        return base_f + merged / count

    return base_f + sum(processed)


def nuslerp(t: float, v0: torch.Tensor, v1: torch.Tensor,
            row_wise: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """NuSLERP: SLERP on normalised inputs, optionally row by row.

    Normalising first handles tensors of different scale better. row_wise
    interpolates each row of a 2D matrix independently, which gives cleaner
    results on attention/MLP weights. Degenerate cases fall back to linear
    interpolation.
    """
    def _slerp_vec(a, b):
        a_n = a / (a.norm() + eps)
        b_n = b / (b.norm() + eps)
        dot = (a_n * b_n).sum().clamp(-1, 1)
        if dot.abs() > 0.9995:
            return (1 - t) * a + t * b
        theta = dot.acos()
        st = theta.sin()
        s0 = (((1 - t) * theta)).sin()
        s1 = ((t * theta)).sin()
        return (s0 / st) * a + (s1 / st) * b

    v0f = v0.float()
    v1f = v1.float()

    if row_wise and v0f.dim() == 2 and v0f.shape == v1f.shape:
        # Vectorised row-wise SLERP (same math as _slerp_vec per row).
        a_n = v0f / (v0f.norm(dim=1, keepdim=True) + eps)
        b_n = v1f / (v1f.norm(dim=1, keepdim=True) + eps)
        dot = (a_n * b_n).sum(dim=1, keepdim=True).clamp(-1, 1)
        theta = dot.acos()
        st = theta.sin()
        s0 = ((1 - t) * theta).sin() / (st + eps)
        s1 = (t * theta).sin() / (st + eps)
        out = s0 * v0f + s1 * v1f
        # Near-parallel rows fall back to lerp; eps only avoids 0/0 on those rows.
        lerp = (1 - t) * v0f + t * v1f
        return torch.where(dot.abs() > 0.9995, lerp, out)

    flat0 = v0f.flatten()
    flat1 = v1f.flatten()
    return _slerp_vec(flat0, flat1).reshape(v0f.shape)


# Gain constant from SuperMerger's trainDifference: alpha=1 matches a 1:1
# training strength.
TRAIN_DIFF_GAIN = 1.8


def train_difference_merge(t_a: torch.Tensor, t_b: torch.Tensor, t_c: torch.Tensor,
                           alpha: float = 1.0) -> torch.Tensor:
    """Train Difference (SuperMerger's `trainDifference`).

    Add Difference scaled by how far A still is from B:

        scale  = |B - A| / (|B - C| + |B - A|)
        result = A + alpha * TRAIN_DIFF_GAIN * scale * (B - C)

    Where A already matches B the update fades out, so a style can be added
    without pulling A toward B. C must be the base B was trained from.
    The reference's sign(B - C) * |scale| * |B - C| form is equivalent, since
    scale is non-negative.
    """
    a = t_a.float()
    b = t_b.float()
    c = t_c.float()

    diff_bc = b - c
    # B == C means nothing was trained, so there is nothing to add.
    if not torch.any(diff_bc):
        return a

    dist_bc = diff_bc.abs()            # how far B moved from C
    dist_ba = (b - a).abs()            # how far A still is from B
    total   = dist_bc + dist_ba
    scale   = torch.where(total != 0, dist_ba / total, torch.zeros_like(total))

    return a + (alpha * TRAIN_DIFF_GAIN) * scale * diff_bc


def model_stock_merge(base: torch.Tensor, others: list) -> torch.Tensor:
    """Model Stock: interpolation ratio from the angle between task vectors.

        cos = mean pairwise cosine similarity between the task vectors
        t   = N * cos / (1 + (N - 1) * cos)
        out = base + t * mean(task vectors)

    Donors that agree (cos -> 1) are averaged in full; unrelated ones (cos -> 0)
    stay near the base. Needs at least two donors (with one it is a plain
    task-vector add). Ignores alpha and block weights.
    """
    b = base.float()
    tvs = [o.float() - b for o in others]
    if not tvs:
        return b
    if len(tvs) == 1:
        # One donor: no angle to measure, plain task-vector add.
        return b + tvs[0]

    n = len(tvs)
    flat = [tv.flatten() for tv in tvs]
    norms = [t.norm() for t in flat]

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            denom = norms[i] * norms[j]
            if float(denom) == 0.0:
                continue
            sims.append(torch.dot(flat[i], flat[j]) / denom)
    if not sims:
        return b

    cos = float(torch.stack(sims).mean())
    denom = 1.0 + (n - 1) * cos
    if abs(denom) < 1e-6:
        # The task vectors cancel out: ratio undefined, keep the base.
        return b
    # Opposed finetunes give cos < 0: clamp t to [0, 1].
    t = min(max((n * cos) / denom, 0.0), 1.0)

    # Accumulate instead of stack().mean() to avoid copying every task vector.
    center = tvs[0].clone()
    for tv in tvs[1:]:
        center += tv
    center /= n
    return b + t * center


# ── Method capability tables ────────────────────────────────────────────────
# Shared by the UI, the input validation and the merge engine.

# Methods that need a third model to define the delta they transfer.
METHODS_REQUIRING_C = ("Add Difference", "Train Difference")

# Donor slots shown in the UI beyond Model B.
EXTRA_DONOR_SLOTS = 3

# Methods that accept any number of donors.
MULTI_DONOR_METHODS = ("Weighted Sum", "TIES", "DARE", "Task Arithmetic",
                       "Breadcrumbs", "DELLA", "Model Stock")

# Methods that build, filter and re-add a task vector (donor - A); Legacy mode
# applies to these. Not the same set as the UI's `is_taskvec` (negative alpha
# allowed), which excludes TIES and DARE.
TASKVEC_METHODS = ("TIES", "DARE", "Task Arithmetic", "Breadcrumbs", "DELLA")

# Methods that derive their own strength and ignore Alpha and block weights.
SELF_WEIGHTING_METHODS = ("Model Stock",)

# Methods that cannot use per-block weights. Empty: in the task-vector methods
# the block weight only scales the vector, and the keep thresholds are relative
# quantiles, so it changes the strength but not which weights are kept.
GLOBAL_ONLY_METHODS: tuple = ()

# ─────────────────────────────────────────────────────────────────────────────

# Gradio inputs taken by the extra donor slots (a dropdown and a weight each).
# They come last in every merge handler's inputs, so the handlers slice them
# off the tail without shifting the block-slider offsets.
_EXTRA_TAIL = 2 * EXTRA_DONOR_SLOTS


def _split_extra_donors(tail):
    """Turn the trailing (dropdowns..., weights...) inputs into (paths, weights).

    Names are resolved with find_model(). Empty or unresolvable slots are
    dropped together with their weight, so each weight stays with its donor.
    """
    tail = list(tail)
    if len(tail) < _EXTRA_TAIL:
        return [], []
    models  = tail[-_EXTRA_TAIL:-EXTRA_DONOR_SLOTS]
    weights = tail[-EXTRA_DONOR_SLOTS:]
    paths, ws = [], []
    for m, w in zip(models, weights):
        if not m:
            continue
        resolved = find_model(m)
        if not resolved:
            _log(f"⚠  Extra donor '{m}' could not be found in the checkpoint "
                 f"folder — skipping that slot.")
            continue
        paths.append(resolved)
        ws.append(float(w) if w is not None else 1.0)
    return paths, ws


_PRECISION_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "fp32": 4}

# Approximate number of copies of the largest tensor alive while one key is
# merged, per method, for N donors. Only used for the RAM warning.
def _peak_multiplier(method: str, n_donors: int) -> int:
    if method == "Model Stock":
        return 2 * n_donors + 2          # task vectors + accumulator
    if method == "TIES":
        return 2 * n_donors + 4          # donors held by caller + task vectors
    if method in ("DARE", "DELLA", "Breadcrumbs", "Task Arithmetic"):
        return n_donors + 5              # these loop one donor at a time
    if method in ("SLERP", "NuSLERP"):
        return 6                         # norms and dot-product temporaries
    if method in ("Add Difference", "Train Difference"):
        return 6
    if method == "Weighted Sum":
        return n_donors + 2
    return n_donors + 5


def estimate_merge_ram(sd_a, keys, precision: str, n_donors: int, method: str):
    """Estimate peak RAM for a merge in bytes: (accumulated output, transient peak).

    The output stays in RAM at the target precision until it is saved (about
    the size of the file). The transient part is several float32 copies of the
    largest single tensor and grows with the donor count. Returns (0, 0) when
    the shapes can't be read; treat that as unknown.
    """
    try:
        out_bytes = 0
        max_elems = 0
        if hasattr(sd_a, "shape_of"):
            for k in keys:
                n = 1
                for d in sd_a.shape_of(k):
                    n *= d
                out_bytes += n
                if n > max_elems:
                    max_elems = n
        else:
            for k in keys:
                n = sd_a[k].numel()
                out_bytes += n
                if n > max_elems:
                    max_elems = n
        out_bytes *= _PRECISION_BYTES.get(precision, 2)
        transient = max_elems * 4 * _peak_multiplier(method, max(1, n_donors))
        return out_bytes, transient
    except Exception:
        return 0, 0


def _method_normalises_donors(method: str, use_ties_variant: bool) -> bool:
    """True when the method divides by the number of contributing donors.

    TIES always does; Breadcrumbs and DELLA only with the sign election on;
    Task Arithmetic and DARE never.
    """
    if method == "TIES":
        return True
    if method in ("Breadcrumbs", "DELLA"):
        return bool(use_ties_variant)
    return False


def donor_weight_advice(method: str, block_weights, donor_weights,
                        use_ties_variant: bool = False):
    """Warning text when the donor weights add up to more than the method can take.

    Each donor is applied with its own weight, so four donors at 0.6 apply 2.4.
    That is fine for methods that divide by the donor count, not for those that
    add. Returns None when there is nothing to report.
    """
    n = len(donor_weights or [])
    if n < 2:
        return None
    if method in SELF_WEIGHTING_METHODS:
        # These ignore the weights.
        return None
    peak  = max(block_weights or [0.0])
    total = peak * (1.0 + sum(float(k) for k in donor_weights[1:]))
    if total <= 1.0:
        return None

    if method == "Weighted Sum":
        return (f"⚠  Donor weights total {total:.2f} at the strongest block, which "
                f"would leave Model A at {1.0 - total:+.2f} — a negative weight, and "
                f"a broken model. The donors will be scaled back to total 1.00 "
                f"automatically. To control this yourself, set Alpha to about "
                f"{1.0 / (n + 1):.2f}, which weighs Model A and all {n} donors "
                f"equally.")
    if not _method_normalises_donors(method, use_ties_variant):
        hint = ("switch on the TIES sign election in the tuning section"
                if method in ("Breadcrumbs", "DELLA") else "use TIES instead")
        return (f"⚠  Donor weights total {total:.2f} at the strongest block. "
                f"{method} adds donors without dividing by how many there are, so "
                f"the result will be over-applied and look saturated. Either lower "
                f"Alpha to about {peak / n:.2f}, or {hint} — those divide by the "
                f"number of donors that actually contributed.")
    return None


def _prefix_of_keys(klist) -> str:
    """Top-level key prefix of a state dict (e.g. "model.diffusion_model.")."""
    for k in klist:
        np_ = _normalise_key(k)
        if np_ != k:
            return k[:len(k) - len(np_)]
    return ""


def _apply_norm_clamp(merged: torch.Tensor, ref: torch.Tensor,
                      strength: float) -> torch.Tensor:
    """Pull a merged tensor's magnitude back toward Model A's.

    Counteracts the norm creep of chained merges (oversaturation, burnt
    contrast) without changing the direction of the change.
    strength 0 = off, 1 = exactly A's norm; values in between interpolate.
    Applied to the whole tensor, not per row, so differences between rows stay.
    """
    if strength <= 0.0:
        return merged
    n_m = merged.norm()
    if not torch.isfinite(n_m) or float(n_m) == 0.0:
        return merged
    n_r = ref.float().norm()
    if not torch.isfinite(n_r) or float(n_r) == 0.0:
        return merged
    target = n_m + (n_r - n_m) * strength
    return merged * (target / n_m)


class _MergeDonor:
    """One donor in a merge: Model B, Model C or an extra donor.

    Holds the lazy state dict, the cross-prefix lookup, the Anima depth remap,
    the synthetic fill for unmapped blocks and per-donor counters.
    `get(key)` takes a Model A key and returns this donor's raw tensor, or None
    when there is no usable counterpart.
    """

    def __init__(self, path, label, sd_a, a_keys, a_prefix, arch,
                 depth_re=None, depth_ns=None, anima_map_source="auto",
                 unmapped_fill="keep_a", allow_fill=True):
        self.path  = path
        self.label = label
        self.sd    = open_model(path, lazy=True)
        self.keys  = list(self.sd.keys())

        # Per-donor diagnostics, aggregated into the merge report at the end.
        self.remap_skipped = 0
        self.synth         = 0

        # Match by key tail when this donor's top-level prefix differs from A's.
        self.lookup = None
        prefix = _prefix_of_keys(self.keys)
        if prefix != a_prefix:
            self.lookup = _build_normalised_lookup(self.keys)
            _log(f"⚠  {label} uses a different key prefix ('{prefix}' vs A's "
                 f"'{a_prefix}'). Matching keys by normalised tail so it is "
                 f"not ignored.")

        self.depth_re  = depth_re
        self.depth_ns  = depth_ns
        self.depth_map = None
        self.fill_map  = {}
        self.fill_mode = "keep_a"
        # 'delta' fill reads Model A and needs the A block matching each donor block.
        self._sd_a   = sd_a
        self.rev_map = {}
        if arch == "anima" and depth_re is not None:
            self._build_depth_map(sd_a, a_keys, anima_map_source,
                                  unmapped_fill if allow_fill else "keep_a")

    # ── setup ───────────────────────────────────────────────────────────────

    def _build_depth_map(self, sd_a, a_keys, anima_map_source, unmapped_fill):
        """Build the block index map when A and this donor are Anima models of
        different depth. Unmatched blocks resolve to None (kept at 100% A)."""
        try:
            n_a = _anima_depth_from_keys(a_keys)
            n_d = _anima_depth_from_keys(self.keys)
            if n_a == n_d:
                return

            _log(f"⚠  Depth mismatch: A has {n_a} DiT blocks, {self.label} has "
                 f"{n_d}. Building block remap from weights...")
            self.depth_map = build_anima_depth_map(sd_a, self.sd, n_a, n_d,
                                                   source=anima_map_source)
            if self.depth_map:
                self.rev_map = {j: i for i, j in self.depth_map.items()
                                if j is not None}
            if anima_map_source == "legacy":
                _log("   Using the LEGACY block map (pre adapter-namespace fix) — "
                     "for reproducing older merges.")
            if not self.depth_map:
                return

            if _LAST_DEPTH_MAP_INFO[0].startswith("derived"):
                _log("   Block map derived from weight similarity "
                     "(no metadata / verified recipe for this pair).")

            new_blocks = sorted(i for i, j in self.depth_map.items() if j is None)
            # Unmatched blocks stay at 100% A unless unmapped_fill builds a
            # stand-in from the neighbouring mapped blocks.
            self.fill_mode = unmapped_fill
            if unmapped_fill in ("nearest", "interpolate", "delta") and new_blocks:
                mapped = sorted(i for i, j in self.depth_map.items() if j is not None)
                for i in new_blocks:
                    lo = max([m for m in mapped if m < i], default=None)
                    hi = min([m for m in mapped if m > i], default=None)
                    d_lo = self.depth_map[lo] if lo is not None else None
                    d_hi = self.depth_map[hi] if hi is not None else None
                    if d_lo is None and d_hi is None:
                        continue
                    if unmapped_fill in ("nearest",) or d_lo is None or d_hi is None:
                        pick = d_lo if d_lo is not None else d_hi
                        self.fill_map[i] = (pick, pick)
                    else:
                        self.fill_map[i] = (d_lo, d_hi)
                _how = ("the donor's CHANGE at the neighbouring block, applied to "
                        "A's own weights"
                        if unmapped_fill == "delta" else
                        "the donor's neighbouring block weights copied in")
                _log(f"   Unmapped-block fill: '{unmapped_fill}' — "
                     f"{len(self.fill_map)} A-only blocks given a synthetic "
                     f"counterpart from {self.label} ({_how}).")
                if unmapped_fill == "interpolate":
                    _log("   ⚠  'interpolate' averages two DIFFERENT layers of the "
                         "donor, which is not a meaningful operator. 'delta' is the "
                         "safer way to let these blocks take part.")
            else:
                _log(f"   Remap built · {n_d} blocks aligned · "
                     f"{len(new_blocks)} A-only blocks kept at 100% A: {new_blocks}")
        except Exception as e:
            _log(f"⚠  Could not build depth remap for {self.label} ({e}) — "
                 f"merging by raw index.")
            self.depth_map = None
            self.fill_map  = {}

    # ── lookup ──────────────────────────────────────────────────────────────

    def _remap_key(self, key):
        """Translate an A key into this donor's equivalent under the depth map.
        Returns None when the block has no counterpart here."""
        if self.depth_map is None or self.depth_re is None:
            return key
        m = self.depth_re.search(key)
        if not m:
            return key
        # Only the main DiT stack is remapped; the adapter's blocks are a
        # separate namespace and keep their own numbering.
        if self.depth_ns is not None and key[:m.start()] != self.depth_ns:
            return key
        j = self.depth_map.get(int(m.group(2)), None)
        if j is None:
            return None
        return key[:m.start()] + m.group(1) + str(j) + m.group(3) + key[m.end():]

    def _raw(self, key):
        if key in self.sd:
            return self.sd[key]
        if self.lookup is not None:
            real = self.lookup.get(_normalise_key(key))
            if real is not None:
                return self.sd[real]
        return None

    def _synthesise(self, key):
        """Stand-in for a block that exists in A but not in this donor, or None.

        'nearest' / 'interpolate' copy the donor's weights from a neighbouring
        block. Risky: that block does a different job in the donor's stack,
        and averaging two different layers is not a meaningful operation.

        'delta' transfers the donor's change instead. For each mapped
        neighbour j, with A_k the block of A that maps to j:

            synthetic = A_i + mean_j(B_j - A_k)

        so the merge lands as A_i + w * mean_j(B_j - A_k) and A's block keeps
        its own scale.
        """
        m = self.depth_re.search(key) if self.depth_re else None
        if not m:
            return None
        if self.depth_ns is not None and key[:m.start()] != self.depth_ns:
            return None
        pair = self.fill_map.get(int(m.group(2)))
        if not pair:
            return None
        d_lo, d_hi = pair

        def _k(j):
            return key[:m.start()] + m.group(1) + str(j) + m.group(3) + key[m.end():]

        if self.fill_mode == "delta":
            return self._synthesise_delta(key, _k, d_lo, d_hi)

        t_lo = self._raw(_k(d_lo))
        if t_lo is None:
            return None
        if d_hi == d_lo:
            return t_lo
        t_hi = self._raw(_k(d_hi))
        if t_hi is None or t_hi.shape != t_lo.shape:
            return t_lo
        if not t_lo.is_floating_point():
            return t_lo
        return ((t_lo.float() + t_hi.float()) * 0.5).to(t_lo.dtype)

    def _synthesise_delta(self, key, _k, d_lo, d_hi):
        """A_i + mean(B_j - A_k) over the mapped neighbours j. See _synthesise."""
        t_a_i = self._sd_a.get(key) if hasattr(self._sd_a, "get") else None
        if t_a_i is None:
            try:
                t_a_i = self._sd_a[key]
            except Exception:
                return None
        if not t_a_i.is_floating_point():
            # No sensible delta for integer buffers; fall back to keeping A.
            return t_a_i

        deltas = []
        for j in ({d_lo, d_hi} - {None}):
            k_a = self.rev_map.get(j)
            if k_a is None:
                continue
            t_b_j = self._raw(_k(j))
            if t_b_j is None or t_b_j.shape != t_a_i.shape:
                continue
            try:
                t_a_k = self._sd_a[_k(k_a)]
            except Exception:
                continue
            if t_a_k.shape != t_a_i.shape:
                continue
            deltas.append(t_b_j.float() - t_a_k.float())

        if not deltas:
            return None
        acc = deltas[0]
        for d in deltas[1:]:
            acc = acc + d
        if len(deltas) > 1:
            acc = acc / len(deltas)
        return (t_a_i.float() + acc).to(t_a_i.dtype)

    def get(self, key):
        """This donor's RAW tensor for A-key `key`, or None."""
        mapped = self._remap_key(key)
        if mapped is None:
            t = self._synthesise(key) if self.fill_map else None
            if t is not None:
                self.synth += 1
                return t
            self.remap_skipped += 1
            return None
        return self._raw(mapped)

    def close(self):
        close_model(self.sd)


def merge_models_method(path_a, path_b, path_c, block_weights_b, global_alpha,
                        method, output_path, precision="fp16", vae_path=None,
                        metadata=None, cb=None,
                        density=0.5, epsilon=0.1, gamma=0.01,
                        nuslerp_row_wise=False, use_ties_variant=False,
                        anima_unmapped_fill="keep_a",
                        anima_map_source="auto",
                        norm_clamp=0.0,
                        donor_weights=None,
                        legacy_taskvec=False):
    """Merge engine for every method.

    path_b:          one donor path or a list of donor paths
    path_c:          Model C, for METHODS_REQUIRING_C
    block_weights_b: per-block weights, used by every method; for the
                     task-vector methods they scale each block's task vector
    global_alpha:    alpha where no per-block weight applies

    Method options:
      density           keep fraction for TIES / DARE / Breadcrumbs / DELLA
      epsilon           DELLA keep-probability spread
      gamma             Breadcrumbs outlier cut
      nuslerp_row_wise  NuSLERP interpolates 2D matrices row by row
      use_ties_variant  Breadcrumbs / DELLA sign election
      norm_clamp        0..1, pull merged norms toward Model A (0 = off)
      donor_weights     multiplier per donor, aligned with path_b (index 0 = 1.0)
      legacy_taskvec    task-vector methods use global_alpha for every block
    """
    import time
    _STOP_MERGE[0] = False
    _QUANTILE_TIME[0] = 0.0
    t_start = time.time()

    arch = detect_arch(path_a)
    _block_index = block_index_fn_for(arch)
    # Fit the weight list to the checkpoint's real depth (deep Anima models have
    # more blocks than the base slider count).
    if arch in DIT_ARCHS:
        _real_blocks = anima_block_count(path_a)
        if len(block_weights_b or []) == len(CKPT_BLOCK_KEYS):
            # SDXL slot list (Easy / Normal): its slots don't exist in a DiT.
            if any(abs(float(w) - float(global_alpha)) > 1e-6
                   for w in block_weights_b[1:]):
                _log(f"⚠  SDXL block weights don't apply to {DIT_ARCH_NAMES[arch]} — "
                     f"using Alpha ({global_alpha:.3f}) for every block. Use the "
                     f"'Anima' per-block mode instead.")
            block_weights_b = [float(global_alpha)] * _real_blocks
        elif _real_blocks != len(block_weights_b or []):
            _log(f"Fitting block weights to {_real_blocks} DiT blocks "
                 f"(list had {len(block_weights_b or [])}).")
            block_weights_b = fit_block_weights(block_weights_b, _real_blocks,
                                                fill=global_alpha)
    _log(f"Starting merge — Method: {method} · Precision: {precision} · "
         f"Architecture: {DIT_ARCH_NAMES.get(arch, arch.upper())}")
    try:
        _src_dt = detect_checkpoint_dtype(path_a)
        if _src_dt == "bf16" and precision == "fp16":
            _log("⚠  Model A is bf16 but the output is fp16. bf16 covers a much wider "
                 "numeric range; clipping it to fp16 can visibly degrade the model, and "
                 "Forge will also run it in fp16. Re-merge with precision 'bf16'.")
    except Exception:
        pass
    # path_b may be a single path or a list of donor paths.
    paths_b = [path_b] if isinstance(path_b, (str, os.PathLike)) else list(path_b)
    paths_b = [p for p in paths_b if p]          # drop empty UI slots
    if not paths_b:
        raise ValueError("No donor model given (path_b is empty).")
    for _pb in paths_b:
        if not os.path.isfile(_pb):
            raise ValueError(
                f"Donor model not found on disk: {_pb!r}. If that looks like a "
                f"bare filename rather than a full path, the name was not "
                f"resolved through find_model() before reaching the engine."
            )
    donor_weights = list(donor_weights or [])
    if len(donor_weights) < len(paths_b):
        donor_weights += [1.0] * (len(paths_b) - len(donor_weights))

    if len(paths_b) > 1 and method not in MULTI_DONOR_METHODS:
        raise ValueError(
            f"{method} is defined for two models only — it cannot take "
            f"{len(paths_b)} donors. Use one of {', '.join(MULTI_DONOR_METHODS)}."
        )
    if method == "Model Stock" and len(paths_b) < 2:
        raise ValueError(
            "Model Stock needs at least TWO donors on top of Model A: it "
            "derives its strength from the angle between their task vectors, "
            "and one vector has no angle to measure."
        )

    _log(f"Model A: {os.path.basename(path_a)}")
    for _i, _pb in enumerate(paths_b):
        _label = "Model B" if len(paths_b) == 1 else f"Donor {_i + 1}"
        _log(f"{_label}: {os.path.basename(_pb)}")
    if path_c:
        _log(f"Model C: {os.path.basename(path_c)}")
    if norm_clamp > 0.0:
        _log(f"Norm clamp: {norm_clamp:.2f} — tensor magnitudes pulled toward Model A.")

    # Lazy open: tensors are read per key.
    if cb: cb(0.05, "Opening Model A...")
    _log("Opening Model A (lazy)...")
    sd_a = open_model(path_a, lazy=True)

    # ── Donors ──────────────────────────────────────────────────────────────
    # One _MergeDonor per model. Model C uses the same class for prefix matching
    # and depth remap.
    if cb: cb(0.25, "Opening Model B...")
    _log("Opening donor model(s) (lazy)...")

    keys   = list(sd_a.keys())
    total  = len(keys)
    result = {}

    a_prefix = _prefix_of_keys(keys)

    # The block regex and adapter namespace come from Model A's key layout.
    _depth_re = None
    _depth_ns = None
    if arch == "anima":
        try:
            import re as _re_dm
            _depth_re = _re_dm.compile(r"((?:^|\.)blocks\.)(\d+)(\.)")
            _depth_ns = anima_block_namespace(keys)
        except Exception as _e:
            _log(f"⚠  Could not prepare the Anima depth regex ({_e}) — "
                 f"merging by raw index.")
            _depth_re = None

    donors = []
    for _i, _pb in enumerate(paths_b):
        donors.append(_MergeDonor(
            _pb,
            "Model B" if len(paths_b) == 1 else f"Donor {_i + 1}",
            sd_a, keys, a_prefix, arch,
            depth_re=_depth_re, depth_ns=_depth_ns,
            anima_map_source=anima_map_source,
            unmapped_fill=anima_unmapped_fill,
        ))

    donor_c = None
    if method in METHODS_REQUIRING_C and path_c:
        if cb: cb(0.35, "Opening Model C...")
        _log("Opening Model C (lazy)...")
        # Model C may synthesise unmapped blocks only with 'delta' fill: B's and
        # C's stand-ins are both anchored on A's block (A_i + X_j - A_k), so B - C
        # reduces to mean_j(B_j - C_j). With 'nearest' / 'interpolate' a synthetic
        # C would invent the baseline the delta is measured against.
        donor_c = _MergeDonor(
            path_c, "Model C", sd_a, keys, a_prefix, arch,
            depth_re=_depth_re, depth_ns=_depth_ns,
            anima_map_source=anima_map_source,
            unmapped_fill=anima_unmapped_fill,
            allow_fill=(anima_unmapped_fill == "delta"),
        )

    _sources = [("Model A", sd_a)] + [(d.label, d.sd) for d in donors]
    if donor_c is not None:
        _sources.append(("Model C", donor_c.sd))
    _bad = [(n, s.unsupported_quant) for n, s in _sources
            if getattr(s, "unsupported_quant", None)]
    _dq = [(n, s.dequantized) for n, s in _sources if getattr(s, "dequantized", 0)]
    if _bad:
        close_model(sd_a)
        for d in donors:
            d.close()
        if donor_c is not None: donor_c.close()
        raise ValueError(
            "Quantized source: " + ", ".join(f"{n} is {'/'.join(f)}" for n, f in _bad)
            + ". Only fp8 checkpoints can be dequantized for merging — merge the "
              "bf16/fp16 releases and quantize the result afterwards.")
    if _dq:
        _log("fp8 source(s) dequantized on read: "
             + ", ".join(f"{n} ({c} tensors)" for n, c in _dq) + ".")
        if precision == "fp8":
            _log("⚠  fp8 output is a plain cast without per-tensor scales — "
                 "save as bf16 for a lossless merge.")

    # Lookups used by the two-model methods.
    _get_b = donors[0].get
    _get_c = (lambda key: None) if donor_c is None else donor_c.get

    if legacy_taskvec:
        if method in TASKVEC_METHODS:
            _log(f"Legacy task-vector mode: {method} uses a single alpha "
                 f"({global_alpha:.3f}) for every block. Per-block weights are "
                 f"ignored — this reproduces builds from before per-block "
                 f"support was added.")
        else:
            _log(f"Legacy task-vector mode is on but has no effect on {method}: "
                 f"it only applies to TIES, DARE, Task Arithmetic, Breadcrumbs "
                 f"and DELLA.")

    _advice = donor_weight_advice(method, block_weights_b, donor_weights,
                                  use_ties_variant)
    if _advice:
        _log(_advice)

    writer = None
    vae_items = None
    if STREAMING_SAVE and isinstance(sd_a, _LazySafetensors) \
            and str(output_path).endswith(".safetensors"):
        try:
            if vae_path:
                vae_items = load_vae_items(vae_path)
            writer = StreamingSafetensorsWriter(
                output_path, plan_streaming_output(sd_a, keys, precision, vae_items),
                metadata)
        except Exception as _e:
            _log(f"⚠  Streaming save unavailable ({_e}) — using the in-memory save.")
            writer = None
            vae_items = None
    if writer is not None:
        import shutil
        try:
            _free = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
        except Exception:
            _free = None
        if _free is not None and _free < writer.total_bytes:
            _need = writer.total_bytes
            writer.abort()
            close_model(sd_a)
            for d in donors:
                d.close()
            if donor_c is not None: donor_c.close()
            raise ValueError(
                f"Not enough disk space for the output: it needs "
                f"{_need / 1024 ** 3:.1f} GB, the target drive has "
                f"{_free / 1024 ** 3:.1f} GB free.")
        _log(f"Streaming save → {os.path.basename(writer.tmp)} "
             f"({writer.total_bytes / 1024 ** 3:.1f} GB, written while merging).")

    _out_b, _tr_b = estimate_merge_ram(sd_a, keys, precision, len(donors), method)
    if writer is not None:
        _out_b = 0
    if _tr_b:
        _gb = 1024 ** 3
        if writer is not None:
            _log(f"RAM estimate: ~{_tr_b / _gb:.1f} GB peak while merging the "
                 f"largest tensor ({len(donors)} donor(s)); the output is streamed "
                 f"to disk.")
        else:
            _log(f"RAM estimate: ~{_out_b / _gb:.1f} GB for the accumulated output "
                 f"+ ~{_tr_b / _gb:.1f} GB peak while merging the largest tensor "
                 f"(~{(_out_b + _tr_b) / _gb:.1f} GB total, {len(donors)} donor(s)).")
        if (_out_b + _tr_b) > 12 * _gb:
            _log("⚠  That is a lot. If this machine has 16 GB or less, reduce the "
                 "donor count, or pick a method that processes donors one at a "
                 "time (DARE, DELLA, Breadcrumbs, Task Arithmetic) instead of "
                 "TIES or Model Stock.")

    if cb: cb(0.50, "Merging...")
    _t_open_end = time.time()   # models opened, merge loop starts

    matched_b = 0
    orphan_b  = 0
    # Keys left at 100% A on purpose by the depth remap, tracked per donor and
    # summed before the report. Not counted as orphans.
    remap_skipped_b = 0
    synth_b = 0
    # Keys present in both models with different shapes.
    shape_skipped = 0
    # Keys where the Weighted Sum donor weights had to be scaled back.
    ws_renormalised = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(keys, desc=f"[NeoMerger] {method}", unit="key",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except ImportError:
        pbar = keys

    def _emit(key, t):
        if writer is not None:
            writer.write(key, t)
        else:
            result[key] = t

    try:
        for ki, key in enumerate(pbar):
            if ki % 100 == 0 and _STOP_MERGE[0]:
                _log("⏹  Merge stopped by user.")
                close_model(sd_a)
                for d in donors:
                    d.close()
                if donor_c is not None: donor_c.close()
                if writer is not None:
                    writer.abort()
                torch.cuda.empty_cache()
                if cb: cb(0.0, "⏹  Stopped by user.")
                return False

            if vae_path and key.startswith(VAE_PREFIX):
                continue

            # ── Weighted Sum, single donor: one lerp in the native dtype ──
            if method == "Weighted Sum" and len(donors) == 1:
                idx = _block_index(key)
                w_b = block_weights_b[idx] if idx < len(block_weights_b) else global_alpha
                t_a_raw = sd_a[key]
                t_b_raw = _get_b(key)
                if t_b_raw is None:
                    orphan_b += 1
                    merged_t = t_a_raw
                else:
                    matched_b += 1
                    if w_b == 0.0 or not t_a_raw.is_floating_point() \
                            or t_a_raw.shape != t_b_raw.shape:
                        merged_t = t_a_raw
                    elif w_b == 1.0:
                        merged_t = t_b_raw
                    else:
                        merged_t = torch.lerp(
                            t_a_raw, t_b_raw.to(dtype=t_a_raw.dtype), float(w_b))
                if norm_clamp > 0.0 and merged_t.is_floating_point() \
                        and merged_t.numel() > 1:
                    merged_t = _apply_norm_clamp(merged_t.float(), t_a_raw, norm_clamp)
                _emit(key, cast_precision(merged_t, precision).contiguous())
                del merged_t
                if cb and ki % 200 == 0:
                    cb(0.5 + 0.40 * (ki / total), f"Keys: {ki}/{total}")
                continue

            # ── Float path ──
            # Non-float tensors (position_ids, masks, ...) keep Model A's tensor.
            _t_a_raw = sd_a[key]
            if not _t_a_raw.is_floating_point():
                _emit(key, _t_a_raw)
                if cb and ki % 200 == 0:
                    cb(0.5 + 0.40 * (ki / total), f"Keys: {ki}/{total}")
                continue

            t_a = _t_a_raw.float()

            # Donors with a usable counterpart for this key. t_b is the first one
            # (two-model methods); t_others is used by the N-donor methods.
            t_others = []
            first_missing = False
            for _di, _d in enumerate(donors):
                _t = _d.get(key)
                if _t is None:
                    if _di == 0:
                        orphan_b += 1
                        first_missing = True
                    continue
                if _di == 0:
                    matched_b += 1
                if _t.shape != _t_a_raw.shape:
                    shape_skipped += 1
                    if _di == 0:
                        first_missing = True
                    continue
                t_others.append(_t.float())

            if not t_others:
                # No donor can contribute here: keep Model A verbatim.
                _emit(key, cast_precision(_t_a_raw, precision).contiguous())
                if cb and ki % 200 == 0:
                    cb(0.5 + 0.40 * (ki / total), f"Keys: {ki}/{total}")
                continue

            # The two-model methods need Model B itself; without it, keep A.
            t_b = t_a if first_missing else t_others[0]

            # ── Per-block weight ──
            idx = _block_index(key)
            w_b = block_weights_b[idx] if idx < len(block_weights_b) else global_alpha

            # A negative alpha on the task-vector methods removes a direction: the
            # block weight sets the magnitude, the global alpha's sign the direction.
            if legacy_taskvec:
                # Legacy: one alpha for the whole model.
                w_tv = global_alpha
            else:
                w_tv = w_b if global_alpha >= 0 else -abs(w_b)

            # Each extra donor's multiplier scales the block weight (Model B = 1.0).
            w_each = [w_tv]
            for _i in range(1, len(t_others)):
                _k = donor_weights[_i] if _i < len(donor_weights) else 1.0
                w_each.append(w_tv * float(_k))

            # Weighted Sum gives A 1 - sum(weights). Above 1.0 that goes negative, so
            # scale the donors back to a total of 1.0, keeping their proportions.
            if method == "Weighted Sum" and len(w_each) > 1:
                _s = sum(w_each)
                if _s > 1.0:
                    w_each = [w / _s for w in w_each]
                    ws_renormalised += 1

            # ── N-donor methods ──
            if method == "Model Stock":
                merged_t = model_stock_merge(t_a, t_others).reshape(t_a.shape)

            elif method == "Weighted Sum" and len(t_others) > 1:
                # N-way average: A keeps whatever weight the donors leave it.
                merged_t = t_a * (1.0 - sum(w_each))
                for _t, _w in zip(t_others, w_each):
                    merged_t = merged_t + _w * _t

            elif method == "TIES" and len(t_others) > 1:
                # The sign election only matters with two or more donors.
                merged_t = ties_merge(
                    [(t_a, _t) for _t in t_others], w_each
                ).reshape(t_a.shape)

            elif method in ("DARE", "Task Arithmetic", "Breadcrumbs", "DELLA") \
                    and len(t_others) > 1:
                if method == "DARE":
                    merged_t = dare_merge(t_a, t_others, w_each)
                elif method == "Task Arithmetic":
                    merged_t = task_arithmetic_merge(t_a, t_others, w_each)
                elif method == "Breadcrumbs":
                    merged_t = breadcrumbs_merge(
                        t_a, t_others, w_each,
                        density=density, gamma=gamma, use_ties=use_ties_variant)
                else:
                    merged_t = della_merge(
                        t_a, t_others, w_each,
                        density=density, epsilon=epsilon, use_ties=use_ties_variant)
                merged_t = merged_t.reshape(t_a.shape)

            elif method == "SLERP":
                if t_a.shape == t_b.shape and t_a.numel() > 1:
                    merged_t = slerp(w_b, t_a, t_b).reshape(t_a.shape)
                else:
                    merged_t = (1 - w_b) * t_a + w_b * t_b

            elif method == "Add Difference":
                # A missing or wrong-shaped C is treated as missing.
                _tc = _get_c(key)
                if _tc is not None and _tc.shape != t_a.shape:
                    shape_skipped += 1
                    _tc = None
                t_c = _tc.float() if _tc is not None else t_a
                merged_t = t_a + w_b * (t_b - t_c)

            elif method == "Train Difference":
                _tc = _get_c(key)
                if _tc is not None and _tc.shape != t_a.shape:
                    shape_skipped += 1
                    _tc = None
                # No counterpart in C: no delta to transfer, keep A.
                if _tc is None:
                    merged_t = t_a
                else:
                    merged_t = train_difference_merge(
                        t_a, t_b, _tc.float(), alpha=w_b
                    ).reshape(t_a.shape)

            elif method == "TIES":
                merged_t = ties_merge(
                    [(t_a, t_b)], [w_tv]
                ).reshape(t_a.shape)

            elif method == "DARE":
                merged_t = dare_merge(t_a, [t_b], [w_tv]).reshape(t_a.shape)

            elif method == "Task Arithmetic":
                merged_t = task_arithmetic_merge(
                    t_a, [t_b], [w_tv]
                ).reshape(t_a.shape)

            elif method == "Breadcrumbs":
                merged_t = breadcrumbs_merge(
                    t_a, [t_b], [w_tv],
                    density=density, gamma=gamma, use_ties=use_ties_variant
                ).reshape(t_a.shape)

            elif method == "DELLA":
                merged_t = della_merge(
                    t_a, [t_b], [w_tv],
                    density=density, epsilon=epsilon, use_ties=use_ties_variant
                ).reshape(t_a.shape)

            elif method == "NuSLERP":
                if t_a.shape == t_b.shape and t_a.numel() > 1:
                    merged_t = nuslerp(w_b, t_a, t_b,
                                       row_wise=nuslerp_row_wise).reshape(t_a.shape)
                else:
                    merged_t = (1 - w_b) * t_a + w_b * t_b

            else:
                # Unknown method: behave like Weighted Sum.
                merged_t = (1 - w_b) * t_a + w_b * t_b

            # Post-merge norm clamp. Applied AFTER the method and BEFORE the
            # precision cast, against Model A's own tensor as the reference.
            if norm_clamp > 0.0 and merged_t.numel() > 1:
                merged_t = _apply_norm_clamp(merged_t, t_a, norm_clamp)

            _emit(key, cast_precision(merged_t, precision).contiguous())
            del t_a, t_b, merged_t

            if cb and ki % 200 == 0:
                cb(0.5 + 0.40 * (ki / total), f"Keys: {ki}/{total}")
    except BaseException:
        if writer is not None:
            writer.abort()
        raise

    close_model(sd_a)
    for d in donors:
        d.close()
    if donor_c is not None: donor_c.close()
    _t_merge_end = time.time()   # merge loop done, VAE swap + save follow

    # Report how much of B contributed (many orphans = the keys didn't line up).
    # Depth-remap skips are intentional and reported separately.
    remap_skipped_b = sum(d.remap_skipped for d in donors)
    synth_b         = sum(d.synth for d in donors)

    real_orphans = max(0, orphan_b - remap_skipped_b)
    if synth_b:
        _log(f"Unmapped-block fill: {synth_b} keys merged against a synthetic "
             f"counterpart interpolated from neighbouring blocks.")
    if remap_skipped_b:
        _log(f"Depth remap: {remap_skipped_b} keys in A-only blocks kept at 100% A "
             f"(expected — these blocks don't exist in Model B).")
    if ws_renormalised:
        _log(f"Donor weights scaled back on {ws_renormalised} keys so Model A "
             f"kept a non-negative weight. Lower Alpha to set the balance "
             f"yourself instead.")
    if shape_skipped:
        _log(f"⚠  Shape mismatch: {shape_skipped} keys exist in both models but with "
             f"different tensor shapes, and were kept at 100% A. If this count is "
             f"high the two models are not the same architecture variant.")
    if matched_b + real_orphans > 0:
        pct_b = 100 * matched_b / (matched_b + real_orphans)
        if real_orphans == 0:
            _log(f"Key match: B contributed to all {matched_b} comparable keys.")
        elif pct_b < 50:
            _log(f"⚠  Key match: B only matched {matched_b}/{matched_b + real_orphans} "
                 f"keys ({pct_b:.0f}%). Most of this merge is just Model A — the two "
                 f"models likely use different module names, not just different "
                 f"prefixes. Run Analyze to see which keys failed to pair.")
        else:
            _log(f"Key match: B matched {matched_b}/{matched_b + real_orphans} keys "
                 f"({pct_b:.0f}%); {real_orphans} A-only keys kept unchanged.")

    if writer is not None:
        if cb: cb(0.95, "Finishing file...")
        try:
            for k, v in (vae_items or []):
                writer.write(k, cast_precision(v, precision))
            vae_items = None
            writer.finish()
        except BaseException:
            writer.abort()
            raise
        _log(f"Saved → {os.path.basename(output_path)} ({precision})")
    else:
        if vae_path:
            if cb: cb(0.90, "Swapping VAE...")
            _log("Swapping VAE...")
            result = swap_vae(result, vae_path)

        if cb: cb(0.95, f"Saving ({precision})...")
        _log(f"Saving → {os.path.basename(output_path)} ({precision})...")
        save_sf(result, output_path, precision=precision, metadata=metadata)
    result = None
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    _t_open  = _t_open_end - t_start
    _t_merge = _t_merge_end - _t_open_end
    _t_save  = time.time() - _t_merge_end
    _log(f"✅ Done in {elapsed:.1f}s → {os.path.basename(output_path)} "
         f"(open {_t_open:.1f}s · merge {_t_merge:.1f}s · save {_t_save:.1f}s"
         f" · quantile {_QUANTILE_TIME[0]:.1f}s)")
    if cb: cb(1.0, "Done!")
    return True

# ── 2. LoRA bake-in ──────────────────────────────────────────────────────────

# Compound module/projection tokens used inside Anima DiT block keys. These keep
# their internal underscore in the checkpoint (e.g. cross_attn.k_proj), so the
# kohya LoRA tail must be reassembled around them instead of split on every '_'.
_ANIMA_COMPOUND_TOKENS = (
    "cross_attn", "self_attn",
    "k_proj", "q_proj", "v_proj", "o_proj", "output_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj", "in_proj",
    "adaln_modulation",
)

def _anima_lora_tail_to_ckpt(tail: str) -> str:
    """
    Convert a kohya LoRA tail (underscore-joined) into the checkpoint's dotted
    module path, preserving compound token names.

    Example:
        'cross_attn_k_proj'      -> 'cross_attn.k_proj'
        'self_attn_output_proj'  -> 'self_attn.output_proj'
        'mlp_layer1'             -> 'mlp.layer1'
        'adaln_modulation_1'     -> 'adaln_modulation.1'

    Strategy: greedily consume known compound tokens from the front; whatever
    isn't a known compound is emitted as a single dotted segment.
    """
    out = []
    i = 0
    s = tail
    while s:
        matched = None
        for tok in _ANIMA_COMPOUND_TOKENS:
            if s == tok or s.startswith(tok + "_"):
                matched = tok
                break
        if matched:
            out.append(matched)
            s = s[len(matched):]
            if s.startswith("_"):
                s = s[1:]
        else:
            # No compound match: take up to the next underscore as one segment.
            head, _, s = s.partition("_")
            out.append(head)
    return ".".join(out)


# Krea 2 diffusers module names -> native, as in Forge Neo's krea2_to_diffusers.
_KREA2_DIFFUSERS_BLOCK = {
    "attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate", "attn.to_out.0": "attn.wo", "attn.to_out": "attn.wo",
    "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
}
_KREA2_DIFFUSERS_STACKS = (
    ("transformer_blocks.", "blocks."),
    ("text_fusion.layerwise_blocks.", "txtfusion.layerwise_blocks."),
    ("text_fusion.refiner_blocks.", "txtfusion.refiner_blocks."),
)
_KREA2_DIFFUSERS_BASIC = {
    "img_in": "first", "time_embed.linear_1": "tmlp.0", "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1", "txt_in.linear_1": "txtmlp.1", "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector", "final_layer.linear": "last.linear",
}


def dit_lora_targets(ckpt_keys, arch: str) -> dict:
    """LoRA module name -> (checkpoint key, qkv third or None) for Krea 2 / MOSS-Image.

    Built from the checkpoint's own keys: native paths, Krea 2 diffusers names
    and MOSS q/k/v projections of a fused qkv, each under the usual prefixes.
    """
    native = {}
    for k in ckpt_keys:
        if k.endswith(".weight"):
            native.setdefault(_normalise_key(k)[:-len(".weight")], k)
    targets = {}

    def add(path, target):
        for name in (path, "diffusion_model." + path, "model.diffusion_model." + path,
                     "transformer." + path, "base_model.model." + path):
            targets.setdefault(name, target)
        flat = path.replace(".", "_")
        for name in ("lora_unet_" + flat, "lora_transformer_" + flat, "lycoris_" + flat):
            targets.setdefault(name, target)

    for p, k in native.items():
        add(p, (k, None))
    if arch == "krea2":
        for p, k in native.items():
            for d_stack, c_stack in _KREA2_DIFFUSERS_STACKS:
                if not p.startswith(c_stack):
                    continue
                n, _, tail = p[len(c_stack):].partition(".")
                for d, c in _KREA2_DIFFUSERS_BLOCK.items():
                    if tail == c:
                        add(f"{d_stack}{n}.{d}", (k, None))
            for d, c in _KREA2_DIFFUSERS_BASIC.items():
                if p == c:
                    add(d, (k, None))
    elif arch == "moss":
        for p, k in native.items():
            if p.endswith(".proj.qkv"):
                for i, proj in enumerate("qkv"):
                    add(p[:-len("qkv")] + proj, (k, i))
    return targets


# Report of the last bake or dry run, read by the UI.
_LAST_BAKE_REPORT = [None]

# LoRA tensor suffixes the bake can't use; listed in the match report.
_UNSUPPORTED_LORA_PATTERNS = (
    ("hada_w",           "LoHa (LyCORIS)"),
    ("lokr_w",           "LoKr (LyCORIS)"),
    ("lora_mid",         "LoCon Tucker/CP (lora_mid)"),
    ("dora_scale",       "DoRA magnitude (dora_scale)"),
    ("oft_",             "OFT / BOFT"),
    ("glora",            "GLoRA"),
    (".lora.down.",      "old Diffusers attn-processor (lora.down/lora.up)"),
    (".lora_A.default.", "PEFT with adapter name (lora_A.default)"),
    (".diff_b",          "bias diff (.diff_b)"),
    (".diff",            "full diff / LyCORIS 'full' (.diff)"),
)


def _classify_unparsed_key(k: str) -> str:
    for pat, label in _UNSUPPORTED_LORA_PATTERNS:
        # .diff / .diff_b are suffixes; matching them as substrings would also
        # hit every 'diffusion_model.' key.
        if pat.startswith(".diff"):
            if k.endswith(pat):
                return label
        elif pat in k:
            return label
    return "unknown format"


def _anima_lora_depth(lora_keys) -> int:
    """Depth of the Anima model a LoRA was trained for, from its keys.

    Handles kohya (lora_unet_blocks_N_...) and dotted (diffusion_model.blocks.N.)
    naming. The highest block index is a lower bound, since a LoRA may skip the
    last blocks, so it is snapped up to the nearest known depth (28, 40, 52).
    """
    import re as _re
    hi = -1
    kre = _re.compile(r"^lora_unet_blocks_(\d+)_")
    for k in lora_keys:
        m = kre.match(k)
        if m:
            hi = max(hi, int(m.group(1)))
    if hi < 0:
        dotted = _anima_depth_from_keys(lora_keys)
        hi = dotted - 1
    seen = hi + 1
    known = sorted({d for pair in ANIMA_KNOWN_DEPTH_MAPS for d in pair}
                   | {ANIMA_NUM_BLOCKS})
    for d in known:
        if d >= seen:
            if d != seen:
                _log(f"   LoRA trains blocks up to {hi}; assuming a {d}-block base "
                     f"(set 'LoRA reference model' if that's wrong).")
            return d
    return seen


def _lokr_factor(lora, base, n):
    """LoKr factor n (1 or 2): full matrix lokr_wN, or lokr_wN_a @ lokr_wN_b.
    Returns (tensor, decomposed_dim or None, keys_used) or (None, None, ()) ."""
    full = f"{base}.lokr_w{n}"
    a, b = f"{base}.lokr_w{n}_a", f"{base}.lokr_w{n}_b"
    if full in lora:
        return lora[full].float(), None, (full,)
    if a in lora and b in lora:
        return lora[a].float() @ lora[b].float(), lora[b].shape[0], (a, b)
    return None, None, ()


def _apply_dora(W: torch.Tensor, delta: torch.Tensor, magnitude: torch.Tensor,
                strength: float, eps: float = 1e-6) -> torch.Tensor:
    """
    DoRA merge (LyCORIS semantics): W_dora = (W + ΔW) * m / ||W + ΔW||, with the
    norm taken per output channel when the magnitude vector matches dim 0, else
    per input channel. The result is blended with the original weight by
    `strength` (the block weight), so strength 0 leaves W untouched.
    """
    Wc = W + delta
    m = magnitude.float()
    tail = [1] * (Wc.dim() - 1)
    if m.numel() == Wc.shape[0]:
        norm = Wc.reshape(Wc.shape[0], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(Wc.shape[0], *tail)
    elif m.numel() == Wc.shape[1]:
        Wt = Wc.transpose(0, 1)
        norm = Wt.reshape(Wt.shape[0], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(Wt.shape[0], *tail).transpose(0, 1)
    else:
        raise ValueError(f"dora_scale {tuple(m.shape)} fits neither axis of {tuple(Wc.shape)}")
    W_dora = Wc * (m.reshape(norm.shape) / (norm + eps))
    if strength == 1.0:
        return W_dora
    return W + strength * (W_dora - W)


def _log_bake_report(r: dict, n_ex: int = 15):
    """Print a readable match report for a bake / dry run."""
    parsed = r["modules_parsed"]
    _log("── LoRA ↔ checkpoint match report ─────────────────────────────")
    _log(f"  arch={r['arch']}  lora tensors={r['lora_tensors']}  "
         f"modules parsed={parsed}  fused={r['fused']}"
         + (f"  ({100*r['fused']/parsed:.1f}%)" if parsed else ""))
    if r["unparsed"]:
        _log(f"  ✗ {sum(len(v) for v in r['unparsed'].values())} LoRA tensors were "
             f"NOT understood by the parser (never considered for baking):")
        for kind, keys in r["unparsed"].items():
            _log(f"     • {kind}: {len(keys)}")
            for k in keys[:n_ex]:
                _log(f"         {k}")
    if r["orphans"]:
        _log(f"  ✗ {len(r['orphans'])} down/A tensors without matching up/B:")
        for k in r["orphans"][:n_ex]:
            _log(f"         {k}")
    labels = {
        "no_mapping":        "no checkpoint key could be built (unsupported naming / TE on Anima / unmapped block)",
        "not_in_checkpoint": "mapped key does NOT exist in the checkpoint",
        "zero_weight":       "block weight is 0 (skipped on purpose)",
        "shape_mismatch":    "key exists but the LoRA shape doesn't fit",
        "unsupported_variant": "recognised format variant not implemented (e.g. Tucker LoKr)",
    }
    for reason, items in r["skipped"].items():
        if not items:
            continue
        _log(f"  ✗ {len(items)} modules — {labels[reason]}:")
        for lk, ck in items[:n_ex]:
            _log(f"         {lk}" + (f"  →  {ck}" if ck else ""))
        if len(items) > n_ex:
            _log(f"         … and {len(items) - n_ex} more")
    if r["peft_alpha_ignored"]:
        _log(f"  ⚠ {r['peft_alpha_ignored']} PEFT modules carry an .alpha that is "
             f"ignored (scale forced to 1.0) — bake strength may be off.")
    if parsed and r["fused"] == parsed and not r["unparsed"]:
        _log("  ✓ Every LoRA module landed on a checkpoint layer.")
    _log("────────────────────────────────────────────────────────────────")


def bake_lora_into_checkpoint(ckpt_path, lora_path, block_weights, output_path,
                              precision="fp16", vae_path=None, save_metadata=True, cb=None,
                              lora_ref_model=None, dry_run=False, report_examples=15):
    """
    Fuses a LoRA into a checkpoint using per-block alpha weights.
    LoRA keys follow the pattern: <name>.lora_up.weight / lora_down.weight / alpha
    The effective delta is: (lora_up @ lora_down) * (alpha / rank) * block_weight

    dry_run=True: opens the checkpoint header only, runs the key mapping and
    shape checks, logs the match report and returns without writing anything.
    The report is also stored in _LAST_BAKE_REPORT[0].
    """
    import time
    _STOP_MERGE[0] = False
    t_start = time.time()
    _log(f"Bake-in: {os.path.basename(lora_path)} → {os.path.basename(ckpt_path)}")

    ckpt_arch = detect_arch(ckpt_path)
    stream = (not dry_run and STREAMING_SAVE and str(ckpt_path).endswith(".safetensors")
              and str(output_path).endswith(".safetensors"))
    if cb: cb(0.05, "Loading checkpoint...")
    _log("Loading checkpoint..." + ("  (dry run — header only)" if dry_run else
                                    "  (lazy, streamed to disk)" if stream else ""))
    sd = open_model(ckpt_path, lazy=True) if (dry_run or stream) else load_sd(ckpt_path)
    stream = stream and isinstance(sd, _LazySafetensors)
    if getattr(sd, "unsupported_quant", None):
        _fmts = "/".join(sd.unsupported_quant)
        close_model(sd)
        raise ValueError(f"The checkpoint is quantized ({_fmts}). Bake into the bf16/fp16 "
                         f"release and quantize the result afterwards.")

    def _shape(k):
        """Target tensor shape; header-only when the checkpoint is lazy."""
        if hasattr(sd, "shape_of"):
            return torch.Size(sd.shape_of(k))
        return sd[k].shape
    _log(f"Architecture: {DIT_ARCH_NAMES.get(ckpt_arch, ckpt_arch.upper())}")
    dit_targets = (dit_lora_targets(list(sd.keys()), ckpt_arch)
                   if ckpt_arch in ("krea2", "moss") else None)

    # Anima block prefix depends on the saver: net., model.diffusion_model. or diffusion_model.
    anima_ckpt_prefix = "net."  # default
    if ckpt_arch == "anima":
        for cand in ("net.blocks.", "model.diffusion_model.blocks.", "diffusion_model.blocks."):
            if any(k.startswith(cand) for k in sd.keys()):
                anima_ckpt_prefix = cand[:-len("blocks.")]  # keep trailing dot, drop 'blocks.'
                break
        _log(f"Anima checkpoint block prefix: {anima_ckpt_prefix}blocks.*")
    if cb: cb(0.30, "Loading LoRA...")
    _log("Loading LoRA...")
    lora = load_sd(lora_path)

    # ── LoRA depth remap ──
    # A 28-block LoRA's indices also exist in a 40/52-block model, so without a
    # remap its deltas land on the wrong layers.
    anima_lora_remap = None
    if ckpt_arch == "anima":
        _ckpt_depth = _anima_depth_from_keys(list(sd.keys()))
        _ref_path = None
        if lora_ref_model:
            _ref_path = find_model(lora_ref_model) or lora_ref_model
        if _ref_path and os.path.isfile(_ref_path):
            # The reference is the LoRA's base model, so its depth is the LoRA's depth.
            _lora_depth = anima_block_count(_ref_path)
        else:
            _lora_depth = _anima_lora_depth(list(lora.keys()))
        if _lora_depth != _ckpt_depth:
            _log(f"⚠  Depth mismatch — LoRA is for a {_lora_depth}-block Anima, "
                 f"checkpoint has {_ckpt_depth} DiT blocks.")
            _m = None
            try:
                if _ref_path and os.path.isfile(_ref_path):
                    _ref_sd = open_model(_ref_path, lazy=True)
                    _m = build_anima_depth_map(sd, _ref_sd, _ckpt_depth, _lora_depth)
                    close_model(_ref_sd)
                else:
                    # No reference: use the checkpoint's metadata or the known
                    # recipes. Can't be verified without the base weights.
                    _meta_src = open_model(ckpt_path, lazy=True) \
                        if _ckpt_depth > _lora_depth else None
                    _deep_map, _how = resolve_depth_map(
                        _meta_src, max(_ckpt_depth, _lora_depth),
                        min(_ckpt_depth, _lora_depth), "auto")
                    close_model(_meta_src)
                    if _deep_map:
                        _log(f"   Block map: {_how} (not verifiable without a "
                             f"reference model).")
                        if _ckpt_depth > _lora_depth:
                            _m = _deep_map                       # ckpt -> lora
                        else:
                            _m = {j: i for i, j in _deep_map.items()
                                  if j is not None}              # ckpt -> lora
                if _m:
                    # invert: LoRA index -> checkpoint index
                    anima_lora_remap = {j: i for i, j in _m.items() if j is not None}
                    _log(f"   Remap ready — {len(anima_lora_remap)} of "
                         f"{_lora_depth} LoRA blocks mapped onto the checkpoint.")
            except Exception as _e:
                _log(f"⚠  Could not build LoRA remap ({_e}).")
            if anima_lora_remap is None:
                _log("   No block map available — indices are applied AS-IS, which "
                     "will land deltas on the wrong layers. Pick the base model this "
                     "LoRA was trained on in 'LoRA reference model' to fix this.")

    if cb: cb(0.50, "Fusing LoRA weights...")
    _log("Fusing LoRA weights...")

    _peft_alpha_ignored = [0]
    unsupported_variant = []      # recognised but not implemented (e.g. Tucker LoKr)
    lora_map = {}
    consumed = set()          # LoRA tensor keys that ended up in lora_map
    orphans  = []             # down/A tensors whose up/B partner is missing
    for key in lora.keys():
        # ── kohya / A1111 naming: lora_down.weight + lora_up.weight (+ optional .alpha) ──
        if "lora_down" in key:
            base = key.replace(".lora_down.weight", "").replace("lora_down.weight", "")
            up_key    = key.replace("lora_down", "lora_up")
            alpha_key = base + ".alpha"
            if up_key not in lora:
                orphans.append(key)
            if up_key in lora:
                consumed.update((key, up_key, alpha_key))
                rank  = lora[key].shape[0]
                alpha = lora[alpha_key].item() if alpha_key in lora else float(rank)
                lora_map[base] = {
                    "down":  lora[key].float(),
                    "up":    lora[up_key].float(),
                    "scale": alpha / rank,
                }
        # ── PEFT / Diffusers naming: lora_A.weight (down) + lora_B.weight (up) ──
        # Used by diffusers-based trainers and ai-toolkit, including native Anima
        # LoRAs. PEFT scale is 1.0.
        elif key.endswith(".lora_A.weight"):
            base = key[:-len(".lora_A.weight")]
            up_key = base + ".lora_B.weight"
            if up_key not in lora:
                orphans.append(key)
            if up_key in lora:
                consumed.update((key, up_key))
                if base + ".alpha" in lora:
                    # Present but ignored below (PEFT path forces scale=1.0).
                    consumed.add(base + ".alpha")
                    _peft_alpha_ignored[0] += 1
                lora_map[base] = {
                    "down":  lora[key].float(),
                    "up":    lora[up_key].float(),
                    "scale": 1.0,
                }

    # ── LyCORIS LoKr: ΔW = kron(w1, w2) * scale ──────────────────────────
    # Each factor is either a full matrix (lokr_wN) or a low-rank pair
    # (lokr_wN_a @ lokr_wN_b). LyCORIS / ComfyUI only apply alpha/dim when a
    # factor is decomposed; with both factors full the scale is 1.0.
    lokr_bases = {k.rsplit(".lokr_w1", 1)[0] for k in lora.keys()
                  if k.endswith(".lokr_w1") or k.endswith(".lokr_w1_a")}
    for base in lokr_bases:
        if f"{base}.lokr_t2" in lora:
            unsupported_variant.append(base)      # Tucker LoKr (conv only)
            continue
        w1, dim1, k1 = _lokr_factor(lora, base, 1)
        w2, dim2, k2 = _lokr_factor(lora, base, 2)
        if w1 is None or w2 is None:
            orphans.append(base + ".lokr_w*")
            continue
        dim = dim2 if dim2 is not None else dim1
        alpha_key = base + ".alpha"
        scale = 1.0
        if dim is not None and alpha_key in lora:
            scale = lora[alpha_key].item() / dim
        lora_map[base] = {"kind": "lokr", "w1": w1, "w2": w2, "scale": scale}
        consumed.update(k1 + k2 + (alpha_key,))

    # ── DoRA magnitude vector, for any module type above ─────────────────
    for base, parts in lora_map.items():
        dk = base + ".dora_scale"
        if dk in lora:
            parts["dora_scale"] = lora[dk].float()
            consumed.add(dk)
    n_dora = sum(1 for p in lora_map.values() if "dora_scale" in p)
    n_lokr = sum(1 for p in lora_map.values() if p.get("kind") == "lokr")
    if n_lokr or n_dora:
        _log(f"  LyCORIS: {n_lokr} LoKr modules, {n_dora} with DoRA magnitude.")

    # ── Keys the parser above did not understand ─────────────────────────
    unparsed = [k for k in lora.keys() if k not in consumed]
    unparsed_by_kind = {}
    for k in unparsed:
        kind = _classify_unparsed_key(k)
        unparsed_by_kind.setdefault(kind, []).append(k)

    # ── Diffusers → kohya naming conversion for SDXL LoRAs ───────────────
    # Diffusers-style SDXL block names (ai-toolkit and other diffusers-based
    # trainers) are converted to kohya names before lookup. SDXL only has
    # attentions in specific positions, so this map is complete.
    DIFFUSERS_TO_KOHYA_SDXL = {
        # Down side: attentions live in input_blocks {4,5} (down_blocks.1) and {7,8} (down_blocks.2)
        "down_blocks_1_attentions_0": "input_blocks_4_1",
        "down_blocks_1_attentions_1": "input_blocks_5_1",
        "down_blocks_2_attentions_0": "input_blocks_7_1",
        "down_blocks_2_attentions_1": "input_blocks_8_1",
        # Down resnets and downsamplers
        "down_blocks_1_resnets_0":     "input_blocks_4_0",
        "down_blocks_1_resnets_1":     "input_blocks_5_0",
        "down_blocks_1_downsamplers_0":"input_blocks_6_0",
        "down_blocks_2_resnets_0":     "input_blocks_7_0",
        "down_blocks_2_resnets_1":     "input_blocks_8_0",
        "down_blocks_0_resnets_0":     "input_blocks_1_0",
        "down_blocks_0_resnets_1":     "input_blocks_2_0",
        "down_blocks_0_downsamplers_0":"input_blocks_3_0",
        # Mid
        "mid_block_attentions_0":      "middle_block_1",
        "mid_block_resnets_0":         "middle_block_0",
        "mid_block_resnets_1":         "middle_block_2",
        # Up side: attentions live in output_blocks {0,1,2} (up_blocks.0) and {3,4,5} (up_blocks.1)
        "up_blocks_0_attentions_0":    "output_blocks_0_1",
        "up_blocks_0_attentions_1":    "output_blocks_1_1",
        "up_blocks_0_attentions_2":    "output_blocks_2_1",
        "up_blocks_1_attentions_0":    "output_blocks_3_1",
        "up_blocks_1_attentions_1":    "output_blocks_4_1",
        "up_blocks_1_attentions_2":    "output_blocks_5_1",
        # Up resnets and upsamplers
        "up_blocks_0_resnets_0":       "output_blocks_0_0",
        "up_blocks_0_resnets_1":       "output_blocks_1_0",
        "up_blocks_0_resnets_2":       "output_blocks_2_0",
        "up_blocks_0_upsamplers_0":    "output_blocks_2_2",
        "up_blocks_1_resnets_0":       "output_blocks_3_0",
        "up_blocks_1_resnets_1":       "output_blocks_4_0",
        "up_blocks_1_resnets_2":       "output_blocks_5_0",
        "up_blocks_1_upsamplers_0":    "output_blocks_5_2",
        "up_blocks_2_resnets_0":       "output_blocks_6_0",
        "up_blocks_2_resnets_1":       "output_blocks_7_0",
        "up_blocks_2_resnets_2":       "output_blocks_8_0",
    }

    _DIFFUSERS_RESNET_TAIL = {
        "norm1":         "in_layers_0",
        "conv1":         "in_layers_2",
        "time_emb_proj": "emb_layers_1",
        "norm2":         "out_layers_0",
        "conv2":         "out_layers_3",
        "conv_shortcut": "skip_connection",
    }

    def _convert_diffusers_lora_key(lora_base: str) -> str:
        """If the LoRA key uses Diffusers naming, rewrite it to kohya naming.
        Returns the unchanged key if no Diffusers pattern is detected."""
        if not lora_base.startswith("lora_unet_"):
            return lora_base
        suffix = lora_base[len("lora_unet_"):]
        # Find the longest matching diffusers prefix
        for diff_prefix, kohya_prefix in DIFFUSERS_TO_KOHYA_SDXL.items():
            if suffix.startswith(diff_prefix + "_"):
                tail = suffix[len(diff_prefix) + 1:]
                # ResNet / downsampler sub-layers have their own names
                # (diffusers conv1 = LDM in_layers.2, ...).
                if "_resnets_" in diff_prefix:
                    tail = _DIFFUSERS_RESNET_TAIL.get(tail, tail)
                elif "_downsamplers_" in diff_prefix and tail == "conv":
                    tail = "op"
                return f"lora_unet_{kohya_prefix}_{tail}"
            if suffix == diff_prefix:
                return f"lora_unet_{kohya_prefix}"
        return lora_base

    def lora_key_to_ckpt(lora_base: str) -> str:
        """
        Convert a LoRA base key to the corresponding checkpoint key.
        Supports SDXL (A1111/Forge native + Diffusers naming) and Anima.
        """
        if dit_targets is not None:
            t = dit_targets.get(lora_base)
            if t is None:
                return ""
            return t[0] if t[1] is None else f"__SLICE3__::{t[1]}::{t[0]}"

        # Normalize Diffusers naming (ai-toolkit etc.) to kohya for SDXL
        lora_base = _convert_diffusers_lora_key(lora_base)

        # Anima UNet: lora_unet_blocks_0_cross_attn_k_proj → net.blocks.0.cross_attn.k_proj.weight
        if ckpt_arch == "anima":
            # PEFT/Diffusers naming: the path after the prefix already matches
            # the checkpoint, so only the prefix is swapped for anima_ckpt_prefix.
            for lora_prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
                if lora_base.startswith(lora_prefix):
                    rest = lora_base[len(lora_prefix):]   # e.g. blocks.0.cross_attn.q_proj  or  llm_adapter.blocks.0...
                    if anima_lora_remap is not None:
                        import re as _re_lr
                        _m2 = _re_lr.match(r"blocks\.(\d+)\.", rest)
                        if _m2:
                            _bn = anima_lora_remap.get(int(_m2.group(1)))
                            if _bn is None:
                                return ""
                            rest = f"blocks.{_bn}." + rest[_m2.end():]
                    return f"{anima_ckpt_prefix}{rest}.weight"
            if lora_base.startswith("lora_unet_blocks_"):
                suffix = lora_base[len("lora_unet_blocks_"):]
                # Split on first _ to get block number
                parts = suffix.split("_", 1)
                if len(parts) == 2:
                    block_n = parts[0]
                    tail = parts[1]
                    # Module names keep their internal underscores (cross_attn,
                    # k_proj, ...), so rebuild the dotted path around them.
                    rest = _anima_lora_tail_to_ckpt(tail)
                    if anima_lora_remap is not None:
                        try:
                            _bn = anima_lora_remap.get(int(block_n))
                        except (TypeError, ValueError):
                            _bn = None
                        if _bn is None:
                            return ""      # LoRA block with no home in this ckpt
                        block_n = _bn
                    return f"{anima_ckpt_prefix}blocks.{block_n}.{rest}.weight"
            # TE keys belong to Qwen3, a separate file: skip
            return ""
        
        k = lora_base

        # SDXL / Illustrious / Pony: A1111/Forge naming already mirrors the
        # checkpoint, so only the separators between parts change.

        def _lora_suffix_to_ckpt(suffix: str) -> str:
            """Convert the suffix of a LoRA key to checkpoint dot-notation.
            Only replaces the separators between logical parts, preserving
            multi-word identifiers like proj_in, input_blocks, text_model."""
            # Known multi-word identifiers that must keep their underscore
            KEEP = [
                "input_blocks", "output_blocks", "middle_block",
                "transformer_blocks", "text_model",
                "proj_in", "proj_out",
                "to_k", "to_q", "to_v",
                "to_out",
                "k_proj", "q_proj", "v_proj", "out_proj",
                "in_proj",
                "in_layers", "out_layers", "emb_layers", "skip_connection",
                "conv_in", "conv_out",
                "time_embed", "label_emb",
                "out_0", "out_1",
                "layer_norm1", "layer_norm2",
                "layer_norm",
                "ln_1", "ln_2", "ln_final", "ln_pre",
                "self_attn", "cross_attn",
                "fc1", "fc2",

            ]
            # Split on '_' and join with '.', keeping known compounds whole.
            parts = suffix.split("_")
            result = []
            i = 0
            while i < len(parts):
                # Try to greedily match known multi-word identifiers
                matched = False
                for length in range(4, 1, -1):
                    candidate = "_".join(parts[i:i+length])
                    if candidate in KEEP:
                        result.append(candidate)
                        i += length
                        matched = True
                        break
                if not matched:
                    result.append(parts[i])
                    i += 1
            return ".".join(result)

        # TE1: CLIP-L
        if k.startswith("lora_te1_text_model_"):
            suffix = k[len("lora_te1_text_model_"):]
            return "conditioner.embedders.0.transformer.text_model." + _lora_suffix_to_ckpt(suffix) + ".weight"

        # TE2: OpenCLIP-G
        # LoRA keys use mlp_fc1/fc2 but OpenCLIP checkpoint uses mlp.c_fc/c_proj
        if k.startswith("lora_te2_text_model_encoder_layers_"):
            suffix = k[len("lora_te2_text_model_encoder_layers_"):]
            # q/k/v are FUSED in the OpenCLIP checkpoint (attn.in_proj_weight).
            # Return a marker the fuse loop resolves to a row-slice bake.
            import re as _re2
            m = _re2.match(r"^(\d+)_self_attn_([qkv])_proj$", suffix)
            if m:
                return f"__TE2_QKV__::{m.group(1)}::{m.group(2)}"
            suffix = _lora_suffix_to_ckpt(suffix)
            # Map fc1→c_fc, fc2→c_proj for OpenCLIP MLP naming
            suffix = suffix.replace("mlp.fc1", "mlp.c_fc").replace("mlp.fc2", "mlp.c_proj")
            # OpenCLIP names the attention module 'attn', not 'self_attn'
            suffix = suffix.replace("self_attn.out_proj", "attn.out_proj")
            return "conditioner.embedders.1.model.transformer.resblocks." + suffix + ".weight"
        if k.startswith("lora_te2_"):
            suffix = k[len("lora_te2_"):]
            return "conditioner.embedders.1.model." + _lora_suffix_to_ckpt(suffix) + ".weight"

        if not k.startswith("lora_unet_"):
            return k + ".weight"

        # UNet: A1111/Forge naming keeps underscores inside block names
        suffix = k[len("lora_unet_"):]
        result = _lora_suffix_to_ckpt(suffix)
        # ff_net.N in LoRA keys = ff.net.N in checkpoint
        import re as _re
        result = _re.sub(r"ff_net_(\d+)", r"ff.net.\1", result)
        result = result.replace("ff_net", "ff.net")
        return "model.diffusion_model." + result + ".weight"

    def _delta(parts, target_shape, full_shape):
        """Full-size ΔW of one LoRA module, before scale; raises when it can't fit."""
        if parts.get("kind", "lora") == "lokr":
            w1, w2 = parts["w1"], parts["w2"]
            if w2.dim() == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            delta = torch.kron(w1, w2)
            if delta.shape != target_shape:
                if delta.numel() != target_shape.numel():
                    raise ValueError(f"kron {tuple(delta.shape)} != target")
                delta = delta.reshape(target_shape)
            return delta
        down, up = parts["down"], parts["up"]
        if down.dim() == 4 and up.dim() == 4:
            # Conv LoRA. kohya convention:
            #   down: [rank, in_ch,  kh, kw]
            #   up:   [out_ch, rank, 1,  1]
            up2 = up.squeeze(-1).squeeze(-1)         # [out_ch, rank]
            rank = up2.shape[1]
            # Handle the inverted-rank variant some trainers produce
            if down.shape[0] != rank and down.shape[1] == rank:
                down = down.transpose(0, 1).contiguous()
            in_ch = down.shape[1]
            kh, kw = down.shape[2], down.shape[3]
            down2 = down.reshape(rank, in_ch * kh * kw)
            out_ch = up2.shape[0]
            delta = (up2 @ down2).reshape(out_ch, in_ch, kh, kw)
            # If the checkpoint key is actually Linear (1x1 squeezed) reshape down
            if delta.shape != target_shape and delta.numel() == full_shape.numel():
                delta = delta.reshape(target_shape)
            if delta.shape != target_shape:
                raise ValueError(f"conv delta {tuple(delta.shape)} != target")
            return delta
        # Linear LoRA: tolerate leading/trailing 1-dims
        up2   = up.squeeze()
        down2 = down.squeeze()
        if up2.dim()   == 1: up2   = up2.unsqueeze(0)
        if down2.dim() == 1: down2 = down2.unsqueeze(1)
        return (up2 @ down2).reshape(target_shape)

    def _apply_op(w_full, op):
        """One fused module on a float copy of its checkpoint tensor."""
        parts, w, qkv_slice, target_shape, full_shape = op
        delta = _delta(parts, target_shape, full_shape)
        W = w_full[qkv_slice[0]:qkv_slice[1]] if qkv_slice is not None else w_full
        dora = parts.get("dora_scale")
        if dora is not None:
            # DoRA: alpha-scaled delta first, block weight blends the result.
            W_new = _apply_dora(W, parts["scale"] * delta, dora, w)
        else:
            scale = parts["scale"] * w
            W_new = W + scale * delta
        if qkv_slice is not None:
            w_full[qkv_slice[0]:qkv_slice[1]] = W_new
            W_new = w_full
        return W_new

    fused = 0
    diffusers_converted = 0  # how many keys were rewritten from Diffusers naming
    skipped = {"no_mapping": [], "not_in_checkpoint": [], "zero_weight": [],
               "shape_mismatch": [],
               "unsupported_variant": [(b, "") for b in unsupported_variant]}
    matched = []
    # checkpoint key -> modules to fuse into it, applied in LoRA order
    ops = {}

    for base_key, parts in lora_map.items():
        # Detect if this key was in Diffusers naming (logged once at the end)
        if base_key.startswith("lora_unet_down_blocks_") or \
           base_key.startswith("lora_unet_up_blocks_") or \
           base_key.startswith("lora_unet_mid_block_"):
            diffusers_converted += 1
        if _STOP_MERGE[0]:
            _log("⏹  Bake-in stopped by user.")
            close_model(sd)
            del sd, lora
            torch.cuda.empty_cache()
            if cb: cb(0.0, "⏹  Stopped by user.")
            return 0
        ckpt_key = lora_key_to_ckpt(base_key)

        # TE2 fused-qkv marker → bake into a row slice of attn.in_proj_weight
        qkv_slice = None
        if ckpt_key.startswith("__TE2_QKV__::"):
            _, layer_n, which = ckpt_key.split("::")
            ckpt_key = (f"conditioner.embedders.1.model.transformer."
                        f"resblocks.{layer_n}.attn.in_proj_weight")
            if ckpt_key in sd:
                d3 = _shape(ckpt_key)[0] // 3
                off = {"q": 0, "k": 1, "v": 2}[which] * d3
                qkv_slice = (off, off + d3)
        # MOSS-Image fused qkv: the LoRA trains q, k or v on its own
        elif ckpt_key.startswith("__SLICE3__::"):
            _, third, ckpt_key = ckpt_key.split("::", 2)
            if ckpt_key in sd:
                d3 = _shape(ckpt_key)[0] // 3
                qkv_slice = (int(third) * d3, (int(third) + 1) * d3)

        if not ckpt_key:
            skipped["no_mapping"].append((base_key, ""))
            continue
        if ckpt_key not in sd:
            skipped["not_in_checkpoint"].append((base_key, ckpt_key))
            continue

        if dit_targets is not None:
            block_idx = get_dit_block_index(ckpt_key, ckpt_arch)
        else:
            block_idx = get_lora_block_index(base_key)
        w = block_weights[block_idx] if block_idx < len(block_weights) else 1.0
        if w < 1e-6:
            skipped["zero_weight"].append((base_key, ckpt_key))
            continue

        kind  = parts.get("kind", "lora")
        down  = parts.get("down")
        up    = parts.get("up")
        full_shape   = _shape(ckpt_key)
        target_shape = full_shape
        if qkv_slice is not None:
            target_shape = torch.Size([qkv_slice[1] - qkv_slice[0],
                                       *full_shape[1:]])

        try:
            _delta(parts, target_shape, full_shape)
        except Exception as e:
            if kind == "lokr":
                src = f"w1={tuple(parts['w1'].shape)} w2={tuple(parts['w2'].shape)}"
            else:
                src = f"up={tuple(up.shape)} down={tuple(down.shape)}"
            skipped["shape_mismatch"].append(
                (base_key, f"{ckpt_key}  {src} target={tuple(target_shape)}"))
            continue

        dora = parts.get("dora_scale")
        if dora is not None and dora.numel() not in (target_shape[0], target_shape[1]):
            skipped["shape_mismatch"].append(
                (base_key, f"{ckpt_key}  dora_scale={tuple(dora.shape)} "
                           f"target={tuple(target_shape)}"))
            continue

        matched.append((base_key, ckpt_key))
        ops.setdefault(ckpt_key, []).append((parts, w, qkv_slice, target_shape, full_shape))
        fused += 1

    if not dry_run and not stream:
        for ckpt_key, key_ops in ops.items():
            for op in key_ops:
                sd[ckpt_key] = cast_precision(_apply_op(sd[ckpt_key].float(), op),
                                              precision).contiguous()

    # ── Match report ─────────────────────────────────────────────────────
    report = {
        "arch":            ckpt_arch,
        "lora_tensors":    len(list(lora.keys())),
        "modules_parsed":  len(lora_map) + len(unsupported_variant),
        "fused":           fused,
        "matched":         matched,
        "skipped":         skipped,
        "unparsed":        unparsed_by_kind,
        "orphans":         orphans,
        "peft_alpha_ignored": _peft_alpha_ignored[0],
    }
    _LAST_BAKE_REPORT[0] = report
    _log_bake_report(report, report_examples)

    if dry_run:
        close_model(sd)
        del sd, lora
        if cb: cb(1.0, f"Dry run: {fused}/{len(lora_map)} modules would be fused.")
        return fused

    if vae_path and not stream:
        if cb: cb(0.88, "Swapping VAE...")
        sd = swap_vae(sd, vae_path)

    meta = None
    if save_metadata:
        meta = {
            "neomerger_version": NEOMERGER_VERSION,
            "merge_type":        "lora_bake",
            "checkpoint":        os.path.basename(ckpt_path),
            "lora":              os.path.basename(lora_path),
            "block_weights":     json.dumps([round(w, 4) for w in block_weights]),
            "precision":         precision,
            "vae":               os.path.basename(vae_path) if vae_path else "original",
            "fused_layers":      str(fused),
        }


    elapsed = time.time() - t_start
    if diffusers_converted > 0:
        _log(f"  → Detected Diffusers-style naming, rewrote {diffusers_converted} keys to kohya layout.")
    if stream:
        if _bake_streamed(sd, ops, _apply_op, output_path, precision, vae_path, meta, cb) is False:
            del sd, lora
            return 0
        elapsed = time.time() - t_start
    else:
        _log(f"Fused {fused} layers. Saving ({precision})...")
        if cb: cb(0.90, f"Fused {fused} LoRA layers. Saving ({precision})...")
        save_sf(sd, output_path, precision=precision, metadata=meta)
    del sd, lora
    torch.cuda.empty_cache()
    _log(f"✅ Done in {elapsed:.1f}s — {fused} layers fused → {os.path.basename(output_path)}")
    if cb: cb(1.0, "Done!")
    return fused

def _bake_streamed(sd, ops, apply_op, output_path, precision, vae_path, meta, cb=None):
    """Write a baked checkpoint tensor by tensor from a lazy source.

    ops: checkpoint key -> modules to fuse (see bake_lora_into_checkpoint).
    Returns False when stopped by the user.
    """
    import shutil
    vae_items = load_vae_items(vae_path) if vae_path else None
    keys = list(sd.keys())
    try:
        writer = StreamingSafetensorsWriter(
            output_path, plan_streaming_output(sd, keys, precision, vae_items), meta)
    except Exception:
        close_model(sd)
        raise
    try:
        _free = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
    except Exception:
        _free = None
    if _free is not None and _free < writer.total_bytes:
        writer.abort()
        close_model(sd)
        raise ValueError(f"Not enough disk space for the output: it needs "
                         f"{writer.total_bytes / 1024 ** 3:.1f} GB, the target drive has "
                         f"{_free / 1024 ** 3:.1f} GB free.")
    _log(f"Fusing and saving → {os.path.basename(writer.tmp)} "
         f"({writer.total_bytes / 1024 ** 3:.1f} GB, {precision})...")
    total = len(keys)
    try:
        for i, k in enumerate(keys):
            if i % 100 == 0 and _STOP_MERGE[0]:
                _log("⏹  Bake-in stopped by user.")
                writer.abort()
                close_model(sd)
                torch.cuda.empty_cache()
                if cb: cb(0.0, "⏹  Stopped by user.")
                return False
            if vae_items is not None and k.startswith(VAE_PREFIX):
                continue
            t = sd[k]
            for op in ops.get(k, ()):
                t = cast_precision(apply_op(t.float(), op), precision)
            writer.write(k, cast_precision(t, precision))
            if cb and i % 200 == 0:
                cb(0.5 + 0.45 * (i / total), f"Keys: {i}/{total}")
        for k, v in (vae_items or []):
            writer.write(k, cast_precision(v, precision))
        close_model(sd)
        writer.finish()
    except BaseException:
        writer.abort()
        close_model(sd)
        raise
    return True

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
            "neomerger_version": NEOMERGER_VERSION,
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

# ── 4. LoRA Extraction  (Checkpoint − Checkpoint → LoRA) ────────────────────

def _ckpt_key_to_lora_base(ckpt_key: str, arch: str, anima_prefix: str = "net."):
    """
    Reverse of lora_key_to_ckpt: convert a checkpoint weight key into the
    kohya-style LoRA base key used when SAVING an extracted LoRA.

    Returns "" for keys that are not standard LoRA targets (norms, embeddings,
    VAE, biases, time embeds, Anima non-block modules, ...).

    SDXL UNet : model.diffusion_model.X.weight
                    → lora_unet_<X with . → _>
    SDXL TE1  : conditioner.embedders.0.transformer.text_model.X.weight
                    → lora_te1_text_model_<X with . → _>
    SDXL TE2  : conditioner.embedders.1.model.transformer.resblocks.N.*  (OpenCLIP)
                    → lora_te2_text_model_encoder_layers_N_*  (HF naming)
                  handled in the extraction loop, because attn.in_proj has
                  to be split into q/k/v (see extract_lora_from_models).
    Anima     : <prefix>blocks.N.rest.weight
                    → lora_unet_blocks_N_<rest with . → _>
                  (round-trips through _anima_lora_tail_to_ckpt in the bake-in)
    """
    if not ckpt_key.endswith(".weight"):
        return ""
    body = ckpt_key[:-len(".weight")]

    if arch == "anima":
        # Only DiT blocks are standard LoRA targets (matches networks.lora_anima
        # and the PEFT/diffusers Anima LoRAs the bake-in supports).
        for pfx in (anima_prefix, "model.diffusion_model.", "diffusion_model.", "net."):
            if body.startswith(pfx + "blocks."):
                rest = body[len(pfx):]                    # blocks.N.cross_attn.k_proj
                return "lora_unet_" + rest.replace(".", "_")
        return ""

    # ── SDXL / Illustrious / Pony ──
    if body.startswith("model.diffusion_model."):
        tail = body[len("model.diffusion_model."):]
        # Skip non-target modules: norms, embeddings, time/label embeds.
        SKIP = ("norm", "emb_layers", "time_embed", "label_emb", "skip_connection")
        if any(s in tail for s in SKIP):
            return ""
        return "lora_unet_" + tail.replace(".", "_")

    if body.startswith("conditioner.embedders.0.transformer.text_model."):
        tail = body[len("conditioner.embedders.0.transformer.text_model."):]
        if "layer_norm" in tail or "embeddings" in tail or "final_layer_norm" in tail:
            return ""
        return "lora_te1_text_model_" + tail.replace(".", "_")

    return ""   # TE2 / VAE / everything else handled (or skipped) by the caller


def _svd_lowrank_pair(delta: torch.Tensor, rank: int, clamp_q: float,
                      device: str = "cpu"):
    """Decompose a 2D delta into (up, down) of at most `rank` via SVD.

        delta ≈ up @ down            up: [out, r] · down: [r, in]

    Singular values are split as sqrt between the two factors, then both are
    clamped at the clamp_q quantile of their combined |values| (kohya-style).
    Non-finite GPU results are recomputed on CPU. Uses the default cusolver
    driver ('gesvd' gives wrong results on wide matrices with some builds).
    Returns (up, down, effective_rank).
    """
    r = int(min(rank, min(delta.shape)))
    if r < 1:
        return None, None, 0
    if not torch.isfinite(delta).all():
        # NaN/Inf in the checkpoint diff: sanitise
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    def _do(dev):
        work = delta.to(device=dev, dtype=torch.float32)
        return torch.linalg.svd(work, full_matrices=False)

    try:
        U, S, Vh = _do(device)
        if device != "cpu" and not (torch.isfinite(U).all()
                                    and torch.isfinite(S).all()
                                    and torch.isfinite(Vh).all()):
            # cusolver returned non-finite values without raising: redo on CPU
            print("[NeoMerger] ⚠️  GPU SVD returned non-finite values — "
                  "recomputed on CPU for this tensor.")
            U, S, Vh = _do("cpu")
    except RuntimeError:
        # CUDA OOM / unsupported driver / solver failure: retry on CPU.
        if device != "cpu":
            U, S, Vh = _do("cpu")
        else:
            raise
    U  = U[:, :r]
    S  = S[:r].clamp(min=0.0)
    Vh = Vh[:r, :]
    sq = torch.sqrt(S)
    up   = U * sq.unsqueeze(0)          # [out, r]
    down = sq.unsqueeze(1) * Vh         # [r, in]
    if clamp_q < 1.0:
        dist = torch.cat([up.flatten(), down.flatten()])
        hi = _quantile_safe(dist, clamp_q)
        up   = up.clamp(-hi, hi)
        down = down.clamp(-hi, hi)
    return up.cpu(), down.cpu(), r


def extract_lora_from_models(path_ft, path_base, output_path, rank=32,
                             conv_rank=0, clamp_q=0.99, include_te=True,
                             precision="fp16", save_metadata=True,
                             use_gpu=True, min_diff=1e-6, cb=None):
    """
    Extracts a LoRA that approximates (finetuned − base):
        base + LoRA@1.0 ≈ finetuned
    Works on SDXL/Illustrious/Pony (UNet + optional TE1/TE2) and Anima
    (DiT blocks). Saved in kohya format (lora_down / lora_up / alpha), so the
    result is directly usable in Forge and in NeoMerger's own Bake-in tab.

      rank      max rank for Linear layers (attention/MLP)
      conv_rank max rank for 3x3 Conv layers; 0 = skip convs (LoCon off)
      clamp_q   outlier clamp quantile applied to the SVD factors
      min_diff  keys whose |delta| never exceeds this are skipped as identical
    """
    import time
    _STOP_MERGE[0] = False
    t_start = time.time()

    arch_ft   = detect_arch(path_ft)
    arch_base = detect_arch(path_base)
    if arch_ft != arch_base:
        raise ValueError(f"Architecture mismatch: finetuned is {arch_ft.upper()}, "
                         f"base is {arch_base.upper()}. Both models must share "
                         f"the same architecture.")
    arch = arch_ft
    if arch in ("krea2", "moss"):
        raise ValueError(f"LoRA extraction is not supported on {DIT_ARCH_NAMES[arch]} "
                         f"models yet — merging and bake-in are.")
    _log(f"LoRA Extraction — {os.path.basename(path_ft)} − {os.path.basename(path_base)}")
    _log(f"Architecture: {arch.upper()} · rank {rank} · conv rank {conv_rank} · clamp {clamp_q}")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        _log("SVD on GPU (per-tensor CPU fallback on OOM).")

    if cb: cb(0.03, "Opening finetuned model...")
    sd_ft = open_model(path_ft, lazy=True)
    if cb: cb(0.08, "Opening base model...")
    sd_bs = open_model(path_base, lazy=True)

    ft_keys = list(sd_ft.keys())

    # Anima models of different depth can't be diffed by key name.
    if arch == "anima":
        _d_ft = _anima_depth_from_keys(ft_keys)
        _d_bs = _anima_depth_from_keys(list(sd_bs.keys()))
        if _d_ft != _d_bs:
            close_model(sd_ft); close_model(sd_bs)
            raise ValueError(
                f"Finetuned model has {_d_ft} DiT blocks, base has {_d_bs}. "
                f"Extract against a base of the same depth (e.g. a 3.8B finetune "
                f"against Anima 3.8B, a 2.9B finetune against Anima 2.9B).")

    # Anima: detect the real top-level prefix of the FINETUNED model.
    anima_prefix = "net."
    if arch == "anima":
        for cand in ("net.blocks.", "model.diffusion_model.blocks.", "diffusion_model.blocks."):
            if any(k.startswith(cand) for k in ft_keys):
                anima_prefix = cand[:-len("blocks.")]
                break
        _log(f"Anima block prefix: {anima_prefix}blocks.*")

    # Cross-prefix matching (same strategy as the merge engine): if base uses a
    # different top-level prefix, match keys by their normalised tail.
    def _prefix_of(klist):
        for k in klist:
            np_ = _normalise_key(k)
            if np_ != k:
                return k[:len(k) - len(np_)]
        return ""
    bs_keylist = list(sd_bs.keys())
    bs_lookup = None
    if _prefix_of(ft_keys) != _prefix_of(bs_keylist):
        bs_lookup = _build_normalised_lookup(bs_keylist)
        _log("⚠  Base model uses a different key prefix — matching by normalised tail.")

    def _get_base(key):
        if key in sd_bs:
            return sd_bs[key]
        if bs_lookup is not None:
            real = bs_lookup.get(_normalise_key(key))
            if real is not None:
                return sd_bs[real]
        return None

    # ── TE2 (OpenCLIP → HF naming) special handling for SDXL ──
    TE2_PREFIX = "conditioner.embedders.1.model.transformer.resblocks."

    def _te2_lora_entries(key: str, delta: torch.Tensor):
        """Yield (lora_base, delta_2d) pairs for a TE2 checkpoint delta.
        attn.in_proj_weight is split into q/k/v; naming converted to HF."""
        tail = key[len(TE2_PREFIX):-len(".weight")] if key.endswith(".weight") \
               else key[len(TE2_PREFIX):]
        # tail examples: '3.attn.in_proj_weight' has no '.weight' suffix; handle both
        if key.endswith(".in_proj_weight"):
            tail = key[len(TE2_PREFIX):-len(".in_proj_weight")]  # 'N.attn' → want N
            layer_n = tail.split(".")[0]
            d = delta.shape[0] // 3
            for name, sl in (("q_proj", slice(0, d)),
                             ("k_proj", slice(d, 2 * d)),
                             ("v_proj", slice(2 * d, 3 * d))):
                yield (f"lora_te2_text_model_encoder_layers_{layer_n}_self_attn_{name}",
                       delta[sl, :])
            return
        parts = tail.split(".")
        layer_n = parts[0]
        rest = ".".join(parts[1:])
        MAP = {
            "attn.out_proj": "self_attn_out_proj",
            "mlp.c_fc":      "mlp_fc1",
            "mlp.c_proj":    "mlp_fc2",
        }
        if rest in MAP:
            yield (f"lora_te2_text_model_encoder_layers_{layer_n}_{MAP[rest]}", delta)

    result = {}
    stats = {"linear": 0, "conv": 0, "te": 0, "skipped_identical": 0,
             "skipped_shape": 0}
    total = len(ft_keys)

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(ft_keys), total=total,
                        desc="[NeoMerger] Extract", unit="key")
    except ImportError:
        iterator = enumerate(ft_keys)

    for ki, key in iterator:
        if ki % 50 == 0:
            if _STOP_MERGE[0]:
                _log("⏹  Extraction stopped by user.")
                close_model(sd_ft); close_model(sd_bs)
                torch.cuda.empty_cache()
                if cb: cb(0.0, "⏹  Stopped by user.")
                return None
            if cb: cb(0.10 + 0.80 * (ki / max(total, 1)),
                      f"Extracting... {ki}/{total} keys")

        is_te2 = arch == "sdxl" and key.startswith(TE2_PREFIX) and \
                 (key.endswith(".weight") or key.endswith(".in_proj_weight"))

        if is_te2:
            if not include_te:
                continue
            lora_base = "__TE2__"   # resolved per-slice below
        else:
            lora_base = _ckpt_key_to_lora_base(key, arch, anima_prefix)
            if not lora_base:
                continue
            if lora_base.startswith("lora_te1_") and not include_te:
                continue

        w_ft = sd_ft[key]
        w_bs = _get_base(key)
        if w_bs is None:
            continue
        if w_ft.shape != w_bs.shape:
            stats["skipped_shape"] += 1
            continue
        if not w_ft.is_floating_point():
            continue

        delta = (w_ft.float() - w_bs.float())
        if delta.abs().max().item() <= min_diff:
            stats["skipped_identical"] += 1
            continue

        try:
            if delta.dim() == 2:
                if is_te2:
                    for lb, sub in _te2_lora_entries(key, delta):
                        up, down, r = _svd_lowrank_pair(sub, rank, clamp_q, device)
                        if r < 1:
                            continue
                        result[lb + ".lora_up.weight"]   = up.contiguous()
                        result[lb + ".lora_down.weight"] = down.contiguous()
                        result[lb + ".alpha"]            = torch.tensor(float(r))
                        stats["te"] += 1
                else:
                    up, down, r = _svd_lowrank_pair(delta, rank, clamp_q, device)
                    if r < 1:
                        continue
                    result[lora_base + ".lora_up.weight"]   = up.contiguous()
                    result[lora_base + ".lora_down.weight"] = down.contiguous()
                    result[lora_base + ".alpha"]            = torch.tensor(float(r))
                    if lora_base.startswith(("lora_te1_", "lora_te2_")):
                        stats["te"] += 1
                    else:
                        stats["linear"] += 1

            elif delta.dim() == 4:
                out_ch, in_ch, kh, kw = delta.shape
                if kh == 1 and kw == 1:
                    # 1x1 conv ≡ Linear
                    up, down, r = _svd_lowrank_pair(
                        delta.reshape(out_ch, in_ch), rank, clamp_q, device)
                    if r < 1:
                        continue
                    result[lora_base + ".lora_up.weight"]   = \
                        up.reshape(out_ch, r, 1, 1).contiguous()
                    result[lora_base + ".lora_down.weight"] = \
                        down.reshape(r, in_ch, 1, 1).contiguous()
                    result[lora_base + ".alpha"]            = torch.tensor(float(r))
                    stats["conv"] += 1
                else:
                    if conv_rank < 1:
                        continue    # LoCon disabled
                    up, down, r = _svd_lowrank_pair(
                        delta.reshape(out_ch, in_ch * kh * kw),
                        conv_rank, clamp_q, device)
                    if r < 1:
                        continue
                    # kohya conv convention: down [r, in, kh, kw] · up [out, r, 1, 1]
                    result[lora_base + ".lora_up.weight"]   = \
                        up.reshape(out_ch, r, 1, 1).contiguous()
                    result[lora_base + ".lora_down.weight"] = \
                        down.reshape(r, in_ch, kh, kw).contiguous()
                    result[lora_base + ".alpha"]            = torch.tensor(float(r))
                    stats["conv"] += 1
        except Exception as e:
            print(f"[NeoMerger] ⚠️  Extraction skipped {key}: {e}")
            continue

    close_model(sd_ft)
    close_model(sd_bs)

    modules_n = sum(1 for k in result if k.endswith(".alpha"))
    if modules_n == 0:
        torch.cuda.empty_cache()
        raise ValueError("No LoRA modules extracted — are the two models actually "
                         "different, and of a supported architecture?")

    meta = None
    if save_metadata:
        from datetime import datetime as _dt
        meta = {
            "neomerger_version":  NEOMERGER_VERSION,
            "extractor_build":    "extract-r2",
            "created":            _dt.now().isoformat(timespec="seconds"),
            "merge_type":         "lora_extract",
            "model_finetuned":    os.path.basename(path_ft),
            "model_base":         os.path.basename(path_base),
            "rank":               str(int(rank)),
            "conv_rank":          str(int(conv_rank)),
            "clamp_quantile":     str(round(float(clamp_q), 4)),
            "include_te":         str(bool(include_te)),
            "precision":          precision,
            # kohya-compatible fields so other UIs display dim/alpha correctly
            "ss_network_module":  "networks.lora_anima" if arch == "anima" else "networks.lora",
            "ss_network_dim":     str(int(rank)),
            "ss_network_alpha":   str(int(rank)),
        }

    if cb: cb(0.92, f"Extracted {modules_n} modules. Saving ({precision})...")
    _log(f"Extracted {modules_n} modules "
         f"(linear {stats['linear']} · conv {stats['conv']} · TE {stats['te']} · "
         f"identical skipped {stats['skipped_identical']}"
         + (f" · shape-mismatch {stats['skipped_shape']}" if stats['skipped_shape'] else "")
         + f"). Saving ({precision})...")
    save_sf(result, output_path, precision=precision, metadata=meta)
    result = None
    torch.cuda.empty_cache()
    elapsed = time.time() - t_start
    _log(f"✅ Done in {elapsed:.1f}s → {os.path.basename(output_path)}")
    if cb: cb(1.0, "Done!")
    return modules_n

# ── 5. Alpha Sweep helpers ───────────────────────────────────────────────────

MAX_SWEEP_STEPS = 16   # hard cap: each step writes a full checkpoint to disk

def _parse_alpha_sweep(spec: str) -> list:
    """
    Parse an alpha sweep spec into a sorted, deduplicated list of floats in [0, 1].

    Accepts comma-separated values and range expressions, freely mixed:
        "0.2, 0.5, 0.8"        → [0.2, 0.5, 0.8]
        "0.2-0.8:0.15"         → [0.2, 0.35, 0.5, 0.65, 0.8]
        "0.1, 0.3-0.5:0.1, 1"  → [0.1, 0.3, 0.4, 0.5, 1.0]
    Raises ValueError on anything unparseable.
    """
    vals = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:   # range; a leading '-' is not a range (and fails float() anyway)
            body, _, step_s = part.partition(":")
            lo_s, _, hi_s = body.partition("-")
            lo, hi = float(lo_s), float(hi_s)
            step = float(step_s) if step_s else 0.1
            if step <= 0 or hi < lo:
                raise ValueError(f"Bad range: '{part}'")
            v = lo
            # tiny epsilon so 0.2-0.8:0.15 includes 0.8 despite float error
            while v <= hi + 1e-9:
                vals.append(round(v, 4))
                v += step
        else:
            vals.append(round(float(part), 4))
    vals = sorted({v for v in vals if 0.0 <= v <= 1.0})
    if not vals:
        raise ValueError("No alpha values in [0, 1] found.")
    if len(vals) > MAX_SWEEP_STEPS:
        raise ValueError(f"{len(vals)} steps requested — max is {MAX_SWEEP_STEPS}. "
                         f"Each step writes a full checkpoint to disk.")
    return vals


def _nm_load_checkpoint_into_ui(ckpt_path: str):
    """Load a just-saved checkpoint into the WebUI (Forge Neo / Forge / A1111).
    Returns None on success, or an error string."""
    sd_models.list_models()
    ckpt_info = sd_models.get_closet_checkpoint_match(os.path.basename(ckpt_path))
    if ckpt_info is None:
        return "model not found by the WebUI after saving"
    if hasattr(sd_models, "forge_model_reload"):
        shared.opts.set("sd_model_checkpoint", ckpt_info.name)
        sd_models.model_data.forge_loading_parameters = dict(
            checkpoint_info=ckpt_info,
            additional_modules=getattr(shared.opts, "forge_additional_modules", []),
            unet_storage_dtype=sd_models.model_data.forge_loading_parameters.get("unet_storage_dtype", None)
                if hasattr(sd_models.model_data, "forge_loading_parameters") else None,
        )
        sd_models.forge_model_reload()
        return None
    if hasattr(sd_models, "reload_model_weights"):
        sd_models.reload_model_weights(info=ckpt_info)
        return None
    return "no compatible model-load API found"


def _nm_generate_preview(prompt, neg, steps, cfg, width, height, seed, sampler):
    """Generate one image with the currently loaded model. Returns a PIL image.
    Raises on failure. Mirrors the Merge & Gen generation path (empty
    ScriptRunner so builtin extension scripts don't crash on missing args)."""
    from contextlib import closing
    import modules.scripts as _scripts_mod

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model       = shared.sd_model,
        outpath_samples= shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids  = shared.opts.outdir_grids   or shared.opts.outdir_txt2img_grids,
        prompt         = prompt,
        negative_prompt= neg,
        sampler_name   = sampler,
        steps          = steps,
        cfg_scale      = cfg,
        width          = width,
        height         = height,
        seed           = seed,
        n_iter         = 1,
        batch_size     = 1,
    )
    try:
        p.scripts = _scripts_mod.ScriptRunner()
        p.script_args = ()
    except Exception:
        p.scripts = None
        p.script_args = ()

    with closing(p):
        processed = processing.process_images(p)
    if not processed or not processed.images:
        raise RuntimeError("generation produced no image")
    return processed.images[0]


def _nm_label_image(img, text):
    """Return a copy of img with a small label box drawn in the top-left."""
    from PIL import ImageDraw, ImageFont
    im = img.copy()
    d = ImageDraw.Draw(im)
    size = max(18, im.width // 28)
    font = None
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * size // 2, size
    pad = size // 3
    d.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0, 200))
    d.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return im


def _nm_make_grid(images):
    """Assemble labeled PIL images into a single comparison grid."""
    from PIL import Image as _PILImage
    import math
    n = len(images)
    cols = min(n, 4) if n > 6 else min(n, 3)
    rows = math.ceil(n / cols)
    w, h = images[0].width, images[0].height
    grid = _PILImage.new("RGB", (cols * w, rows * h), (24, 24, 24))
    for i, im in enumerate(images):
        if im.size != (w, h):
            im = im.resize((w, h))
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid

# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPERS  (shared slider blocks)
# ═════════════════════════════════════════════════════════════════════════════

def build_anima_easy_sliders(prefix=""):
    """Semantic sliders for Anima's DiT blocks. Returns a dict."""
    sliders = {}
    for cat_name, cat_info in ANIMA_EASY_CATEGORIES.items():
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
    return sliders


def build_easy_sliders(prefix=""):
    """Returns (sliders_dict, use_exp_checkbox)."""
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
        use_exp = gr.Checkbox(
            value=False,
            label="Include experimental sliders in the merge",
            info="When off, experimental sliders are ignored entirely so they don't dilute the main categories.",
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

    return sliders, use_exp

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

                            # Extra donors, in a closed accordion; EXTRA_DONOR_SLOTS sets the count.
                            with gr.Accordion("➕  More donors  (optional)",
                                              open=False,
                                              visible=False) as bm_extra_donor_group:
                                gr.Markdown(
                                    "<small>Model B is already donor 1 — these are "
                                    "extra ones. Leave blank to ignore. Each weight "
                                    "multiplies that donor's share of the block "
                                    "weight, so 1.0 means the same strength as "
                                    "Model B.<br><br>Donors should all be finetunes "
                                    "of the <b>same base</b> as Model A: the merge "
                                    "works on <code>donor − A</code>, and if a donor "
                                    "has a different origin that difference is noise "
                                    "rather than training.</small>"
                                )
                                bm_extra_models  = []
                                bm_extra_weights = []
                                for _slot in range(EXTRA_DONOR_SLOTS):
                                    with gr.Row():
                                        bm_extra_models.append(gr.Dropdown(
                                            ml, value=None,
                                            label=f"Donor {_slot + 2}", scale=3))
                                        bm_extra_weights.append(gr.Slider(
                                            0.0, 2.0, value=1.0, step=0.05,
                                            label="weight", scale=1))

                            bm_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                            bm_refresh.click(
                                fn=lambda: [gr.update(choices=get_model_list())]
                                           * (3 + EXTRA_DONOR_SLOTS),
                                outputs=[bm_model_a, bm_model_b, bm_model_c]
                                        + bm_extra_models
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

                        # Quick shortcuts — set up a style add/remove in one click
                        with gr.Group():
                            gr.Markdown("### ⚡ Quick actions")
                            gr.Markdown(
                                "<small>Pick **Model A** (your model) and **Model B** (the style), "
                                "then click one. You can still tweak everything below before merging.</small>"
                            )
                            with gr.Row():
                                bm_quick_add    = gr.Button("➕  Add style",    variant="secondary", scale=1)
                                bm_quick_remove = gr.Button("🚫  Remove style", variant="secondary", scale=1)

                        gr.Markdown("")

                        # Merge method
                        with gr.Group():
                            gr.Markdown("### ⚗️ Merge Method")
                            bm_method = gr.Radio(
                                ["Weighted Sum", "SLERP", "Add Difference",
                                 "Train Difference", "TIES", "DARE",
                                 "Task Arithmetic", "Breadcrumbs", "DELLA",
                                 "NuSLERP", "Model Stock"],
                                value="Weighted Sum", label="",
                            )
                            bm_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                label="Alpha (global weight)",
                                info="Used by all methods. For block merge, overridden per-block "
                                     "— on the task-vector methods a negative alpha still sets "
                                     "the direction while the block sliders set the strength.")
                            bm_method_info = gr.Markdown(
                                "<small>**Weighted Sum:** `(1-α)·A + α·B` — standard linear blend.</small>"
                            )
                            bm_legacy_taskvec = gr.Checkbox(
                                value=False, visible=False,
                                label="Legacy mode  (one Alpha, no per-block weights)",
                                info="How this method behaved before it supported per-block "
                                     "weights: a single Alpha for the whole model. Use it to "
                                     "reproduce an older recipe, or to check whether a "
                                     "difference you are seeing comes from the block weights. "
                                     "Switching it on locks the per-block checkbox off. It does "
                                     "not restore the dtype and shape-mismatch fixes — those "
                                     "were bugs, not behaviour.",
                            )

                            # Only "Filter strength" is up front. allow_negative and
                            # use_ties_variant are hidden state kept for the click() wiring.
                            bm_allow_negative = gr.Checkbox(value=False, visible=False)
                            bm_use_ties_variant = gr.Checkbox(value=False, visible=False)

                            bm_density = gr.Slider(
                                0.1, 1.0, value=0.7, step=0.05, visible=False,
                                label="Filter strength",
                                info="Higher = keep more of the model's changes · "
                                     "lower = filter more aggressively. 0.7 is a good default.",
                            )

                            # Method-specific tuning, shown only for the selected method.
                            with gr.Accordion("⚙️ Method tuning (optional)",
                                              open=False, visible=False) as bm_method_tuning:
                                bm_epsilon = gr.Slider(
                                    0.0, 0.5, value=0.1, step=0.01, visible=False,
                                    label="DELLA — magnitude bias (epsilon)",
                                    info="How strongly DELLA keeps big changes over small ones. Default 0.1 is fine.",
                                )
                                bm_gamma = gr.Slider(
                                    0.0, 0.2, value=0.01, step=0.005, visible=False,
                                    label="Breadcrumbs — outlier cut (gamma)",
                                    info="How much of the largest outliers to drop. Default 0.01 is fine.",
                                )
                                bm_nuslerp_rowwise = gr.Checkbox(
                                    value=False, visible=False,
                                    label="NuSLERP — row-wise interpolation",
                                    info="Cleaner on attention/MLP matrices, slightly slower.",
                                )

                        gr.Markdown("")

                        # Depth-upscale settings apply to every Anima merge, so they
                        # live outside the Anima block panel.
                        with gr.Group(visible=False) as bm_upscale_panel:
                            gr.Markdown("### 🪜 Depth-Upscale  <small style='color:gray'>"
                                        "(Model A and B have different block counts)</small>")
                            bm_anima_fill = gr.Dropdown(
                                ["keep_a", "delta", "nearest", "interpolate"],
                                value="keep_a",
                                label="Blocks missing from Model B",
                                info="keep_a = leave them at 100% Model A (safe, but those "
                                     "blocks don't take part) · delta = transfer the donor's "
                                     "CHANGE at the neighbouring block, keeping A's own weights "
                                     "and scale (recommended when you want them to take part) · "
                                     "nearest = copy the donor's neighbouring block wholesale — "
                                     "that block does a different job in A's stack · "
                                     "interpolate = average two of the donor's neighbouring "
                                     "blocks; averaging two different layers is not a valid "
                                     "operation, kept only to reproduce old merges."
                            )
                            bm_anima_map = gr.Dropdown(
                                ["auto", "known", "measured", "legacy"],
                                value="auto",
                                label="Block map source",
                                info="auto = verified recipe, falling back to measurement · "
                                     "measured = always derive from the weights · "
                                     "legacy = the pre-fix map (blocks 2-7 differ); only for "
                                     "reproducing merges made before the adapter fix."
                            )
                            bm_upscale_info = gr.Markdown("")

                        # Block merge toggle
                        with gr.Group():
                            gr.Markdown("### 🧱 Block Merge")
                            bm_use_blocks = gr.Checkbox(
                                value=False, label="Enable per-block weights",
                                info="Override alpha for individual blocks using Easy or Normal mode"
                            )
                            # Only for GLOBAL_ONLY_METHODS (currently none).
                            bm_blocks_disabled_msg = gr.Markdown(
                                "<small style='color:orange'>This method works globally and "
                                "ignores per-block weights.</small>",
                                visible=False
                            )

                        gr.Markdown("")

                        # Output
                        with gr.Group():
                            gr.Markdown("### 💾 Output")
                            bm_out_name = gr.Textbox(label="File name", value="",
                                                     placeholder=_NAME_PLACEHOLDER)
                            gr.Markdown("<small>Saved as `.safetensors` in your checkpoints folder. Existing files are kept: a new one gets `_2`, `_3`, … (change in Settings → NeoMerger).</small>")

                        gr.Markdown("")

                        # Advanced Options
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            bm_precision = gr.Radio(
                                ["fp16", "bf16", "fp32", "fp8"], value="fp16",
                                label="Output precision",
                                info="Match the source model's dtype. bf16 and fp16 are "
                                     "both 16-bit but split the bits differently: saving a "
                                     "bf16 model as fp16 clips its range and degrades it. "
                                     "fp32 = full quality (2x size) · fp8 = experimental"
                            )

                            def _bm_suggest_precision(model_a, current):
                                """Warn when the chosen output dtype would narrow the
                                source model's range (bf16 -> fp16 is lossy in a way
                                that fp16 -> bf16 is not)."""
                                try:
                                    pa = find_model(model_a) if model_a else None
                                    src = detect_checkpoint_dtype(pa) if pa else None
                                except Exception:
                                    src = None
                                if src == "bf16" and current == "fp16":
                                    return gr.update(value=(
                                        "<small style='color:orange'>⚠ Model A is <b>bf16</b>; "
                                        "saving as fp16 clips its numeric range and can visibly "
                                        "degrade output. Switch to <b>bf16</b>.</small>"))
                                return gr.update(value="")

                            bm_precision_warn = gr.Markdown("")
                            bm_model_a.change(_bm_suggest_precision,
                                              inputs=[bm_model_a, bm_precision],
                                              outputs=[bm_precision_warn])
                            bm_precision.change(_bm_suggest_precision,
                                                inputs=[bm_model_a, bm_precision],
                                                outputs=[bm_precision_warn])
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
                            bm_norm_clamp = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.05,
                                label="Norm clamp (magnitude drift)",
                                info="Pulls each merged tensor's magnitude back toward "
                                     "Model A. 0 = off. Counteracts the oversaturation "
                                     "that builds up over chained merges, without "
                                     "weakening the change itself the way a lower alpha "
                                     "does. Try 0.2-0.4 on a model merged several times.",
                            )
                            bm_save_meta = gr.Checkbox(value=True, label="Save merge metadata")

                        gr.Markdown("")

                        # Presets
                        (bm_p_name, bm_p_save, bm_p_dd,
                         bm_p_load, bm_p_del, bm_p_status) = build_preset_row("bm_")

                        gr.Markdown("")
                        with gr.Row():
                            bm_run_btn       = gr.Button("🔀  Run Merge",   variant="primary",   scale=2)
                            bm_mergegen_btn  = gr.Button("🎨  Merge & Gen", variant="secondary", scale=2, elem_id="neomerger_mergegen_btn")
                            bm_stop_btn      = gr.Button("⏹  Stop",          variant="stop",      scale=1)
                        bm_status   = gr.Markdown("")
                        bm_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress", interactive=False)

                        # Hidden fields populated by JS from the txt2img tab
                        bm_t2i_prompt = gr.Textbox(visible=False, elem_id="neomerger_t2i_prompt")
                        bm_t2i_neg    = gr.Textbox(visible=False, elem_id="neomerger_t2i_neg")
                        bm_t2i_steps  = gr.Number(visible=False, value=28, elem_id="neomerger_t2i_steps")
                        bm_t2i_cfg    = gr.Number(visible=False, value=7.0, elem_id="neomerger_t2i_cfg")
                        bm_t2i_w      = gr.Number(visible=False, value=1024, elem_id="neomerger_t2i_w")
                        bm_t2i_h      = gr.Number(visible=False, value=1024, elem_id="neomerger_t2i_h")
                        bm_t2i_seed   = gr.Number(visible=False, value=-1,   elem_id="neomerger_t2i_seed")
                        bm_t2i_sampler= gr.Textbox(visible=False, value="Euler a", elem_id="neomerger_t2i_sampler")

                        # Result preview (Supermerger-style — inside the tab)
                        with gr.Accordion("🖼️  Generated Preview", open=True, visible=False) as bm_gen_panel:
                            bm_gen_image = gr.Image(label="Merge result", type="pil", interactive=False)
                            bm_gen_info  = gr.Textbox(label="Generation info", lines=3, interactive=False)

                        gr.Markdown("")

                        # Alpha Sweep — merge a series of alphas and compare
                        with gr.Accordion("🧪  Alpha Sweep  (Merge & Gen series)", open=False):
                            gr.Markdown(
                                "<small>Runs the current merge setup at **several alpha values**, "
                                "generating one image per alpha with the **same seed** and your "
                                "current txt2img settings, then builds a labeled comparison grid.<br>"
                                "The sweep varies the <b>global alpha</b> — per-block weights are "
                                "ignored. Each step writes a full checkpoint to disk; intermediates "
                                "are deleted afterwards unless you keep them.</small>"
                            )
                            bm_sweep_spec = gr.Textbox(
                                label="Alpha values", value="0.2-0.8:0.15",
                                info="Comma list and/or ranges: '0.2, 0.5, 0.8' · '0.2-0.8:0.15' · mixed. Max 16 steps."
                            )
                            bm_sweep_keep = gr.Checkbox(
                                value=False, label="Keep intermediate models",
                                info="Keep every <name>_aX.XX.safetensors in the checkpoints folder. "
                                     "Off = all sweep files are deleted after the grid is built."
                            )
                            bm_sweep_btn = gr.Button("🧪  Run Alpha Sweep", variant="secondary",
                                                     elem_id="neomerger_sweep_btn")
                            with gr.Group(visible=False) as bm_sweep_panel:
                                bm_sweep_grid    = gr.Image(label="Comparison grid", type="pil",
                                                            interactive=False)
                                bm_sweep_gallery = gr.Gallery(label="Individual results",
                                                              columns=4, object_fit="contain",
                                                              height=320)

                    with gr.Column(scale=2):

                        with gr.Group(visible=False) as bm_blocks_panel:
                            with gr.Group():
                                gr.Markdown("### ⚙️ Block Mode")
                                bm_mode = gr.Radio(
                                    ["Easy", "Normal", "Anima Easy", "Anima"], value="Easy", label="",
                                    info="Easy / Normal = SDXL · Anima Easy = semantic DiT · Anima = per-block DiT (Anima, Krea 2, MOSS-Image)"
                                )

                            gr.Markdown("")

                            with gr.Group(visible=True) as bm_easy_panel:
                                gr.Markdown("### 🎛️ Semantic Sliders")
                                gr.Markdown("<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B</small>")
                                gr.Markdown("")
                                bm_easy_sliders, bm_use_exp = build_easy_sliders("bm_")
                                gr.Markdown("")
                                bm_e2n_btn = gr.Button(
                                    "➡️  Convert to Normal",
                                    size="sm", variant="secondary"
                                )
                                gr.Markdown(
                                    "<small>Expands the semantic sliders into the 20 per-block "
                                    "weights and switches to Normal mode for fine-tuning.</small>"
                                )

                            with gr.Group(visible=False) as bm_anima_easy_panel:
                                gr.Markdown("### 🎛️ Semantic Sliders — Anima")
                                gr.Markdown(
                                    "<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B &nbsp;·&nbsp; "
                                    "mapped onto Anima's DiT blocks</small>"
                                )
                                gr.Markdown(
                                    "<small style='color:gray'>Layer roles from Team-C's Anima "
                                    "per-layer breakdown. Categories overlap (a block can serve "
                                    "two roles) — overlapping blocks are averaged.</small>"
                                )
                                gr.Markdown("")
                                bm_anima_easy_sliders = build_anima_easy_sliders("bm_")
                                gr.Markdown("")
                                bm_ae2b_btn = gr.Button(
                                    "➡️  Convert to per-block",
                                    size="sm", variant="secondary"
                                )
                                gr.Markdown(
                                    "<small style='color:gray'>Writes the computed weights into "
                                    "the Anima per-block sliders so you can fine-tune them.</small>"
                                )

                            with gr.Group(visible=False) as bm_norm_panel:
                                gr.Markdown("### 🧱 Per-Block Weights — SDXL")
                                gr.Markdown("<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B</small>")
                                gr.Markdown("")
                                bm_block_sliders = build_block_sliders(0.5)

                            with gr.Group(visible=False) as bm_anima_panel:
                                gr.Markdown("### 🧱 Per-Block Weights — DiT (Anima / Krea 2 / MOSS-Image)")
                                bm_anima_count_md = gr.Markdown(
                                    "<small>**0** = 100% Model A &nbsp;·&nbsp; **1** = 100% Model B &nbsp;·&nbsp; "
                                    f"{ANIMA_NUM_BLOCKS} DiT blocks detected</small>"
                                )
                                gr.Markdown(
                                    "<small style='color:orange'>Block labels are positional only — "
                                    "no documentation exists yet on what each block controls visually.</small>"
                                )
                                gr.Markdown("")
                                bm_anima_sliders = {}
                                with gr.Row():
                                    anima_btn_a   = gr.Button("All → A (0.0)", size="sm", variant="secondary")
                                    anima_btn_mid = gr.Button("All → 0.5",     size="sm", variant="secondary")
                                    anima_btn_b   = gr.Button("All → B (1.0)", size="sm", variant="secondary")
                                gr.Markdown("")
                                # Gradio can't add components later: build ANIMA_MAX_BLOCKS
                                # sliders and show as many as Model A has blocks.
                                for i in range(ANIMA_MAX_BLOCKS):
                                    bm_anima_sliders[f"anima_block_{i}"] = gr.Slider(
                                        0.0, 1.0, value=0.5, step=0.05, label=f"BLK_{i:02d}",
                                        visible=(i < ANIMA_NUM_BLOCKS)
                                    )
                                asv = list(bm_anima_sliders.values())
                                anima_btn_a.click(fn=lambda: [gr.update(value=0.0)] * len(asv), outputs=asv)
                                anima_btn_b.click(fn=lambda: [gr.update(value=1.0)] * len(asv), outputs=asv)
                                anima_btn_mid.click(fn=lambda: [gr.update(value=0.5)] * len(asv), outputs=asv)

                                def _anima_sync_depth(model_a):
                                    """Show exactly as many block sliders as Model A has
                                    DiT blocks. Falls back to the base 28 when the model
                                    isn't Anima or can't be read."""
                                    n = ANIMA_NUM_BLOCKS
                                    try:
                                        if model_a:
                                            pa = find_model(model_a)
                                            if pa and detect_arch(pa) in DIT_ARCHS:
                                                n = anima_block_count(pa)
                                    except Exception:
                                        n = ANIMA_NUM_BLOCKS
                                    n = max(1, min(int(n), ANIMA_MAX_BLOCKS))
                                    note = ""
                                    if n != ANIMA_NUM_BLOCKS:
                                        note = (" &nbsp;·&nbsp; <span style='color:orange'>"
                                                f"depth-upscaled ({n - ANIMA_NUM_BLOCKS:+d} vs base)</span>")
                                    md = ("<small>**0** = 100% Model A &nbsp;·&nbsp; "
                                          "**1** = 100% Model B &nbsp;·&nbsp; "
                                          f"{n} DiT blocks detected{note}</small>")
                                    return [gr.update(visible=(i < n))
                                            for i in range(ANIMA_MAX_BLOCKS)] + [gr.update(value=md)]

                                bm_model_a.change(
                                    fn=_anima_sync_depth,
                                    inputs=[bm_model_a],
                                    outputs=asv + [bm_anima_count_md],
                                )

                                def _sync_upscale_panel(model_a, model_b):
                                    """Show the depth-upscale controls only when A and B
                                    really differ in block count — that's the only case
                                    where they do anything."""
                                    na = nb = None
                                    try:
                                        if model_a:
                                            pa = find_model(model_a)
                                            if pa and detect_arch(pa) in DIT_ARCHS:
                                                na = anima_block_count(pa)
                                        if model_b:
                                            pb = find_model(model_b)
                                            if pb and detect_arch(pb) in DIT_ARCHS:
                                                nb = anima_block_count(pb)
                                    except Exception:
                                        na = nb = None
                                    if na and nb and na != nb:
                                        msg = (f"<small style='color:orange'>Model A has "
                                               f"<b>{na}</b> DiT blocks, Model B has <b>{nb}</b>. "
                                               f"{na - nb} block(s) of A have no counterpart in B — "
                                               f"the setting above decides what happens to them. "
                                               f"Applies to every merge method, per-block weights "
                                               f"or not.</small>")
                                        return gr.update(visible=True), gr.update(value=msg)
                                    return gr.update(visible=False), gr.update(value="")

                                for _c in (bm_model_a, bm_model_b):
                                    _c.change(
                                        fn=_sync_upscale_panel,
                                        inputs=[bm_model_a, bm_model_b],
                                        outputs=[bm_upscale_panel, bm_upscale_info],
                                    )

                            bm_mode.change(
                                fn=lambda m: (
                                    gr.update(visible=m=="Easy"),
                                    gr.update(visible=m=="Normal"),
                                    gr.update(visible=m=="Anima"),
                                    gr.update(visible=m=="Anima Easy"),
                                ),
                                inputs=bm_mode,
                                outputs=[bm_easy_panel, bm_norm_panel, bm_anima_panel,
                                         bm_anima_easy_panel]
                            )

                            def bm_anima_easy_to_blocks(model_a, *easy_vals):
                                """Compute per-block weights from the Anima semantic
                                sliders, push them into the Anima per-block sliders,
                                and switch the Mode radio over."""
                                n = _anima_ui_depth(model_a)
                                bw = anima_easy_to_block_weights(
                                    dict(zip(bm_anima_easy_sliders.keys(), easy_vals)), n)
                                return ([gr.update(value=round(bw[i], 3))
                                         if i < len(bw) else gr.update()
                                         for i in range(ANIMA_MAX_BLOCKS)]
                                        + [gr.update(value="Anima")])

                            def bm_easy_to_normal(use_exp, *easy_vals):
                                """Compute the 20 per-block weights from the current
                                Easy sliders, push them into the Normal sliders, and
                                switch the Mode radio to Normal."""
                                slider_dict = dict(zip(bm_easy_sliders.keys(), easy_vals))
                                block_weights = easy_to_block_weights(slider_dict, use_experimental=bool(use_exp))
                                return (
                                    [gr.update(value=round(w, 4)) for w in block_weights]
                                    + [gr.update(value="Normal")]
                                )

                            bm_e2n_btn.click(
                                fn=bm_easy_to_normal,
                                inputs=[bm_use_exp] + list(bm_easy_sliders.values()),
                                outputs=list(bm_block_sliders.values()) + [bm_mode],
                            )

                        with gr.Group(visible=True) as bm_no_blocks_panel:
                            gr.Markdown("### ℹ️ Block merge disabled")
                            gr.Markdown(
                                "Enable **per-block weights** on the left to control how each "
                                "part of the model blends. When disabled, the global **Alpha** "
                                "slider is used for the UNet and the **Base Alpha** slider below "
                                "is used for the base block (text encoder · CLIP)."
                            )
                            gr.Markdown("")
                            bm_base_alpha = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Base Alpha (text encoder · CLIP)",
                                info="Controls the base block only. Useful for TIES/DARE where "
                                     "per-block weights are not available.",
                            )

                METHOD_INFO = {
                    "Weighted Sum":   "<small>**Weighted Sum:** `(1-α)·A + α·B` — standard linear blend. Most compatible.</small>",
                    "SLERP":          "<small>**SLERP:** Spherical interpolation — smoother transitions, fewer artifacts at extreme alpha values.</small>",
                    "Add Difference": "<small>**Add Difference:** `A + α·(B-C)` — transfers the difference between B and C onto A. Requires Model C.</small>",
                    "Train Difference": "<small>**Train Difference:** `A + α·1.8·scale·(B-C)`, where `scale` fades the update out wherever A already matches B. Pushes a style onto A without pulling A's identity toward B — closer to continued finetuning than to mixing. Requires Model C (must be the base B was trained from).</small>",
                    "TIES":           "<small>**TIES:** Trims the smallest changes before merging — cleaner results. Note: sign election needs 2+ donors, so with a single Model B this is effectively trim-and-add.</small>",
                    "DARE":           "<small>**DARE:** Drops and rescales weights randomly — reduces merge noise.</small>",
                    "Task Arithmetic": "<small>**Task Arithmetic:** `A + α·(B-A)` as a task vector. With **negative α** it SUBTRACTS B's direction (remove a style). Unfiltered — best on clean models.</small>",
                    "Breadcrumbs":    "<small>**Breadcrumbs:** Task arithmetic that drops BOTH big outliers (gamma) and small noise, keeping the middle band. Best for chained-merge / 'dirty' checkpoints.</small>",
                    "DELLA":          "<small>**DELLA:** DARE with magnitude-ranked drop — keeps important changes more often.</small>",
                    "NuSLERP":        "<small>**NuSLERP:** Normalised SLERP with optional row-wise interpolation — cleaner transitions than plain SLERP.</small>",
                    "Model Stock":    "<small>**Model Stock:** Derives the blend strength from the ANGLE between the donors' task vectors instead of an alpha — donors that agree are taken in full, donors that disagree are trusted less. **Ignores Alpha and the block weights by design.** Needs at least TWO donors on top of Model A, all finetuned from the same base.</small>",
                }

                def _blocks_disabled_reason(method, legacy):
                    """Message explaining why per-block weights are off, or None
                    when they are available. Single source of truth for the
                    checkbox state, the panel visibility and the notice text."""
                    if bool(legacy) and method in TASKVEC_METHODS:
                        return ("<small style='color:orange'>Legacy mode is on: this "
                                "method uses a single Alpha for the whole model, so "
                                "per-block weights are switched off.</small>")
                    if method in SELF_WEIGHTING_METHODS:
                        return ("<small style='color:orange'>This method derives its "
                                "own blend strength from the donors, so Alpha and the "
                                "per-block weights have no effect.</small>")
                    if method in GLOBAL_ONLY_METHODS:
                        return ("<small style='color:orange'>This method works globally "
                                "and ignores per-block weights.</small>")
                    return None

                def on_legacy_toggle(legacy, method):
                    """Legacy mode ignores the per-block weights entirely, so lock the
                    checkbox off instead of leaving it live and inert."""
                    reason = _blocks_disabled_reason(method, legacy)
                    block_ok = reason is None
                    return (
                        gr.update(interactive=block_ok, value=False),  # bm_use_blocks
                        gr.update(visible=False),                      # bm_blocks_panel
                        gr.update(visible=True),                       # bm_no_blocks_panel
                        gr.update(visible=not block_ok,
                                  value=reason or ""),                 # bm_blocks_disabled_msg
                    )

                def on_method_change(method, legacy=False):
                    show_c        = method in METHODS_REQUIRING_C
                    show_extra    = method in MULTI_DONOR_METHODS
                    self_weighted = method in SELF_WEIGHTING_METHODS
                    # A self-weighting method derives its own strength, so both
                    # the alpha and the block sliders are inert for it.
                    alpha_ok      = not self_weighted
                    disabled_msg  = _blocks_disabled_reason(method, legacy)
                    block_ok      = disabled_msg is None
                    info          = METHOD_INFO.get(method, "")

                    # Which extra controls each method needs (front panel = minimal).
                    is_taskvec    = method in ("Task Arithmetic", "Breadcrumbs", "DELLA")
                    show_density  = method in ("Breadcrumbs", "DELLA")   # "Filter strength"
                    show_epsilon  = method == "DELLA"                    # in tuning accordion
                    show_gamma    = method == "Breadcrumbs"              # in tuning accordion
                    show_rowwise  = method == "NuSLERP"                  # in tuning accordion
                    show_legacy   = method in TASKVEC_METHODS   # own checkbox, top level
                    # The accordion must be visible whenever any of its children is.
                    show_tuning   = show_epsilon or show_gamma or show_rowwise

                    # Task-vector methods can SUBTRACT a style, so the alpha slider
                    # opens up to -1..+1 automatically (no checkbox to discover).
                    alpha_min = -1.0 if is_taskvec else 0.0

                    return (
                        gr.update(visible=show_c),               # bm_model_c_row
                        gr.update(interactive=block_ok),         # bm_use_blocks
                        gr.update(visible=not block_ok,
                                  value=disabled_msg or ""),     # bm_blocks_disabled_msg
                        gr.update(value=info),                   # bm_method_info
                        gr.update(visible=False),                # bm_blocks_panel (reset)
                        gr.update(visible=True),                 # bm_no_blocks_panel (reset)
                        gr.update(value=False),                  # bm_use_blocks reset
                        gr.update(),                             # bm_allow_negative (hidden no-op)
                        gr.update(visible=show_density),         # bm_density
                        gr.update(visible=show_epsilon),         # bm_epsilon
                        gr.update(visible=show_gamma),           # bm_gamma
                        gr.update(),                             # bm_use_ties_variant (hidden no-op)
                        gr.update(visible=show_rowwise),         # bm_nuslerp_rowwise
                        gr.update(visible=show_tuning),          # bm_method_tuning
                        gr.update(visible=show_legacy),          # bm_legacy_taskvec
                        gr.update(visible=show_extra),           # bm_extra_donor_group
                        # bm_alpha stays LAST: _quick_setup overrides upd[-1].
                        gr.update(minimum=alpha_min, interactive=alpha_ok),
                    )

                bm_method.change(
                    fn=on_method_change,
                    inputs=[bm_method, bm_legacy_taskvec],
                    outputs=[bm_model_c_row, bm_use_blocks, bm_blocks_disabled_msg,
                             bm_method_info, bm_blocks_panel, bm_no_blocks_panel, bm_use_blocks,
                             bm_allow_negative, bm_density, bm_epsilon, bm_gamma,
                             bm_use_ties_variant, bm_nuslerp_rowwise, bm_method_tuning,
                             bm_legacy_taskvec, bm_extra_donor_group, bm_alpha]
                )

                # --- Quick action shortcuts: Add / Remove style ---
                # Both pick Breadcrumbs (good on chained-merge models) and set a
                # sensible alpha. The user can still change anything afterwards.
                def _quick_setup(alpha_value):
                    # Reuse on_method_change for Breadcrumbs to refresh all panels,
                    # then override the alpha slider's value on top.
                    upd = list(on_method_change("Breadcrumbs", legacy=False))
                    # Last item is the bm_alpha update — set both range and value.
                    upd[-1] = gr.update(minimum=-1.0, maximum=1.0, value=alpha_value)
                    # Prepend the method radio update so the UI shows "Breadcrumbs".
                    return [gr.update(value="Breadcrumbs")] + upd

                _quick_outputs = [
                    bm_method,
                    bm_model_c_row, bm_use_blocks, bm_blocks_disabled_msg,
                    bm_method_info, bm_blocks_panel, bm_no_blocks_panel, bm_use_blocks,
                    bm_allow_negative, bm_density, bm_epsilon, bm_gamma,
                    bm_use_ties_variant, bm_nuslerp_rowwise, bm_method_tuning,
                    bm_legacy_taskvec, bm_extra_donor_group, bm_alpha,
                ]

                bm_quick_add.click(
                    fn=lambda: _quick_setup(0.5),
                    outputs=_quick_outputs,
                )
                bm_quick_remove.click(
                    fn=lambda: _quick_setup(-0.3),
                    outputs=_quick_outputs,
                )

                def on_blocks_toggle(enabled, method, legacy):
                    if _blocks_disabled_reason(method, legacy) is not None:
                        return gr.update(visible=False), gr.update(visible=True)
                    return gr.update(visible=enabled), gr.update(visible=not enabled)

                bm_use_blocks.change(
                    fn=on_blocks_toggle,
                    inputs=[bm_use_blocks, bm_method, bm_legacy_taskvec],
                    outputs=[bm_blocks_panel, bm_no_blocks_panel]
                )

                bm_legacy_taskvec.change(
                    fn=on_legacy_toggle,
                    inputs=[bm_legacy_taskvec, bm_method],
                    outputs=[bm_use_blocks, bm_blocks_panel,
                             bm_no_blocks_panel, bm_blocks_disabled_msg]
                )

                bm_all = (list(bm_easy_sliders.values()) +
                          list(bm_block_sliders.values()) +
                          list(bm_anima_sliders.values()) +
                          list(bm_anima_easy_sliders.values()))

                # Wire the "Convert to per-block" button now that the Anima
                # per-block sliders (asv) and the mode radio both exist.
                bm_ae2b_btn.click(
                    fn=bm_anima_easy_to_blocks,
                    inputs=[bm_model_a] + list(bm_anima_easy_sliders.values()),
                    outputs=asv + [bm_mode],
                )

                def bm_run(method, alpha, base_alpha, use_blocks, mode, ma, mb, mc,
                           out_n, precision, vae, save_meta, use_exp,
                           density, epsilon, gamma, use_ties_variant,
                           nuslerp_rowwise, allow_negative, anima_fill, anima_map,
                           norm_clamp, legacy_taskvec, *vals):
                    extra_mb, extra_w = _split_extra_donors(vals)
                    vals = vals[:-_EXTRA_TAIL] if len(vals) >= _EXTRA_TAIL else vals
                    if not ma or not mb:
                        return "❌  Select both Model A and Model B.", 0.0
                    if method in METHODS_REQUIRING_C and not mc:
                        return f"❌  {method} requires Model C.", 0.0

                    # Methods that never use per-block weights.
                    NO_BLOCK = GLOBAL_ONLY_METHODS

                    pa = find_model(ma)
                    pb = find_model(mb)
                    pc = find_model(mc) if mc else None
                    vp = find_vae(vae) if vae and vae != "— keep original —" else None
                    op = resolve_output_path(
                        get_ckpt_dir(), out_n,
                        auto_merge_name(method, alpha, ma, mb, mc, extra_mb,
                                        use_blocks and method not in NO_BLOCK),
                        inputs=[pa, pb, pc, *extra_mb])

                    ne = len(bm_easy_sliders)
                    nn = len(bm_block_sliders)
                    na = len(bm_anima_sliders)
                    nae = len(bm_anima_easy_sliders)

                    if use_blocks and method not in NO_BLOCK:
                        if mode == "Easy":
                            bw = easy_to_block_weights(dict(zip(bm_easy_sliders.keys(), vals[:ne])), use_experimental=bool(use_exp))
                        elif mode == "Anima Easy":
                            bw = anima_easy_to_block_weights(
                                dict(zip(bm_anima_easy_sliders.keys(),
                                         vals[ne + nn + na: ne + nn + na + nae])),
                                _anima_ui_depth(ma))
                        elif mode == "Anima":
                            bw = [float(v) if v is not None else 0.5
                                  for v in vals[ne + nn: ne + nn + na]]
                        else:
                            bw = [float(v) if v is not None else 0.5 for v in vals[ne:ne+nn]]
                    else:
                        # No per-block weights: use global alpha for UNet,
                        # but override the base (text encoder) block with base_alpha.
                        if mode in ("Anima", "Anima Easy"):
                            bw = [float(alpha)] * _anima_ui_depth(ma)
                        else:
                            bw = [float(alpha)] * 20
                            bw[0] = float(base_alpha)   # index 0 = "base"

                    meta = None
                    if save_meta:
                        meta = {
                            "neomerger_version": NEOMERGER_VERSION,
                            "merge_type":        "block_merge",
                            "merge_method":      method,
                            "model_a":           os.path.basename(pa),
                            "model_b":           os.path.basename(pb),
                            "model_c":           os.path.basename(pc) if pc else "N/A",
                            "alpha":             str(round(float(alpha), 4)),
                            "base_alpha":        str(round(float(base_alpha), 4)),
                            "block_weights":     json.dumps([round(w, 4) for w in
                                                             _meta_trim_bw(bw, mode, ma)]),
                            "precision":         precision,
                            "vae":               os.path.basename(vp) if vp else "original",
                        }
                        # Depth-upscale settings change the output: record them.
                        if str(mode).startswith("Anima"):
                            meta["anima_map_source"]     = str(anima_map)
                            meta["anima_unmapped_fill"]  = str(anima_fill)
                            meta["anima_blocks"]         = str(_anima_ui_depth(ma))
                        # Record extra params only for the methods that use them.
                        if method in ("Breadcrumbs", "DELLA"):
                            meta["density"] = str(round(float(density), 4))
                        if method == "DELLA":
                            meta["epsilon"] = str(round(float(epsilon), 4))
                        if method == "Breadcrumbs":
                            meta["gamma"] = str(round(float(gamma), 4))
                        if method in ("Breadcrumbs", "DELLA"):
                            meta["ties_variant"] = str(bool(use_ties_variant))
                        if method == "NuSLERP":
                            meta["nuslerp_row_wise"] = str(bool(nuslerp_rowwise))
                        if float(norm_clamp) > 0.0:
                            meta["norm_clamp"] = str(round(float(norm_clamp), 4))
                        if bool(legacy_taskvec):
                            meta["legacy_taskvec"] = "True"

                    try:
                        result = merge_models_method(
                            pa, [pb] + extra_mb, pc, bw, float(alpha),
                            method, op, precision, vp, meta,
                            density=float(density), epsilon=float(epsilon),
                            gamma=float(gamma),
                            nuslerp_row_wise=bool(nuslerp_rowwise),
                            use_ties_variant=bool(use_ties_variant),
                            anima_unmapped_fill=anima_fill,
                            anima_map_source=anima_map,
                            norm_clamp=float(norm_clamp),
                            donor_weights=[1.0] + extra_w,
                            legacy_taskvec=bool(legacy_taskvec),
                        )
                        if result is False:
                            return "⏹  Merge stopped by user.", 0.0
                        return f"✅  Saved to `{op}`", 1.0
                    except Exception as e:
                        return f"❌  {e}", 0.0

                def bm_mergegen(method, alpha, base_alpha, use_blocks, mode, ma, mb, mc,
                                out_n, precision, vae, save_meta, use_exp,
                                t2i_prompt, t2i_neg, t2i_steps, t2i_cfg,
                                t2i_w, t2i_h, t2i_seed, t2i_sampler,
                                density, epsilon, gamma, use_ties_variant,
                                nuslerp_rowwise, allow_negative, anima_fill, anima_map,
                                norm_clamp, legacy_taskvec, *vals):
                    """Merge + generate test image using current txt2img settings."""
                    extra_mb, extra_w = _split_extra_donors(vals)
                    vals = vals[:-_EXTRA_TAIL] if len(vals) >= _EXTRA_TAIL else vals
                    if not ma or not mb:
                        return ("❌  Select both Model A and Model B.", 0.0,
                                gr.update(visible=False), None, "")
                    if method in METHODS_REQUIRING_C and not mc:
                        return (f"❌  {method} requires Model C.", 0.0,
                                gr.update(visible=False), None, "")

                    pa = find_model(ma)
                    pb = find_model(mb)
                    pc = find_model(mc) if mc else None
                    vp = find_vae(vae) if vae and vae != "— keep original —" else None
                    op = resolve_output_path(
                        get_ckpt_dir(), out_n,
                        auto_merge_name(method, alpha, ma, mb, mc, extra_mb,
                                        use_blocks and method not in GLOBAL_ONLY_METHODS),
                        inputs=[pa, pb, pc, *extra_mb])

                    ne = len(bm_easy_sliders)
                    nn = len(bm_block_sliders)
                    na = len(bm_anima_sliders)
                    nae = len(bm_anima_easy_sliders)

                    if use_blocks and method not in GLOBAL_ONLY_METHODS:
                        if mode == "Easy":
                            bw = easy_to_block_weights(dict(zip(bm_easy_sliders.keys(), vals[:ne])), use_experimental=bool(use_exp))
                        elif mode == "Anima Easy":
                            bw = anima_easy_to_block_weights(
                                dict(zip(bm_anima_easy_sliders.keys(),
                                         vals[ne + nn + na: ne + nn + na + nae])),
                                _anima_ui_depth(ma))
                        elif mode == "Anima":
                            bw = [float(v) if v is not None else 0.5
                                  for v in vals[ne + nn: ne + nn + na]]
                        else:
                            bw = [float(v) if v is not None else 0.5 for v in vals[ne:ne + nn]]
                    else:
                        if mode in ("Anima", "Anima Easy"):
                            bw = [float(alpha)] * _anima_ui_depth(ma)
                        else:
                            bw = [float(alpha)] * 20
                            bw[0] = float(base_alpha)

                    meta = None
                    if save_meta:
                        meta = {
                            "neomerger_version": NEOMERGER_VERSION,
                            "merge_type":        "block_merge",
                            "merge_method":      method,
                            "model_a":           os.path.basename(pa),
                            "model_b":           os.path.basename(pb),
                            "model_c":           os.path.basename(pc) if pc else "N/A",
                            "alpha":             str(round(float(alpha), 4)),
                            "base_alpha":        str(round(float(base_alpha), 4)),
                            "block_weights":     json.dumps([round(w, 4) for w in
                                                             _meta_trim_bw(bw, mode, ma)]),
                            "precision":         precision,
                            "vae":               os.path.basename(vp) if vp else "original",
                        }
                        # Depth-upscale settings change the output: record them.
                        if str(mode).startswith("Anima"):
                            meta["anima_map_source"]     = str(anima_map)
                            meta["anima_unmapped_fill"]  = str(anima_fill)
                            meta["anima_blocks"]         = str(_anima_ui_depth(ma))
                        if method in ("Breadcrumbs", "DELLA"):
                            meta["density"] = str(round(float(density), 4))
                        if method == "DELLA":
                            meta["epsilon"] = str(round(float(epsilon), 4))
                        if method == "Breadcrumbs":
                            meta["gamma"] = str(round(float(gamma), 4))
                        if method in ("Breadcrumbs", "DELLA"):
                            meta["ties_variant"] = str(bool(use_ties_variant))
                        if method == "NuSLERP":
                            meta["nuslerp_row_wise"] = str(bool(nuslerp_rowwise))
                        if float(norm_clamp) > 0.0:
                            meta["norm_clamp"] = str(round(float(norm_clamp), 4))
                        if bool(legacy_taskvec):
                            meta["legacy_taskvec"] = "True"

                    try:
                        _log("Merge & Gen — starting merge step...")
                        result = merge_models_method(
                            pa, [pb] + extra_mb, pc, bw, float(alpha),
                            method, op, precision, vp, meta,
                            density=float(density), epsilon=float(epsilon),
                            gamma=float(gamma),
                            nuslerp_row_wise=bool(nuslerp_rowwise),
                            use_ties_variant=bool(use_ties_variant),
                            anima_unmapped_fill=anima_fill,
                            anima_map_source=anima_map,
                            norm_clamp=float(norm_clamp),
                            donor_weights=[1.0] + extra_w,
                            legacy_taskvec=bool(legacy_taskvec),
                        )
                        if result is False:
                            return ("⏹  Merge stopped by user.", 0.0,
                                    gr.update(visible=False), None, "")
                    except Exception as e:
                        return (f"❌  Merge failed: {e}", 0.0,
                                gr.update(visible=False), None, "")

                    try:
                        _log("Merge & Gen — loading merged model into WebUI...")
                        sd_models.list_models()
                        ckpt_info = sd_models.get_closet_checkpoint_match(os.path.basename(op))
                        if ckpt_info is None:
                            return (f"✅  Merge saved to `{op}` but model not found for generation.",
                                    1.0, gr.update(visible=False), None, "")

                        # Forge Neo / Forge: switch via forge_loading_parameters + forge_model_reload
                        if hasattr(sd_models, "forge_model_reload"):
                            shared.opts.set("sd_model_checkpoint", ckpt_info.name)
                            sd_models.model_data.forge_loading_parameters = dict(
                                checkpoint_info=ckpt_info,
                                additional_modules=getattr(shared.opts, "forge_additional_modules", []),
                                unet_storage_dtype=sd_models.model_data.forge_loading_parameters.get("unet_storage_dtype", None)
                                    if hasattr(sd_models.model_data, "forge_loading_parameters") else None,
                            )
                            sd_models.forge_model_reload()
                        # A1111 / classic Forge fallback
                        elif hasattr(sd_models, "reload_model_weights"):
                            sd_models.reload_model_weights(info=ckpt_info)
                        else:
                            return (f"✅  Merge saved to `{op}` but no compatible model-load API found.",
                                    1.0, gr.update(visible=False), None, "")
                    except Exception as e:
                        return (f"✅  Merge saved to `{op}`\n❌  Model load failed: {e}",
                                1.0, gr.update(visible=False), None, "")

                    try:
                        _log("Merge & Gen — generating preview image...")
                        from contextlib import closing
                        import modules.scripts as _scripts_mod

                        prompt = (t2i_prompt or "").strip()
                        if not prompt:
                            prompt = "masterpiece, best quality, 1girl, portrait"
                            _log("  → No prompt set in txt2img; using fallback prompt.")

                        try:
                            seed_val = int(float(t2i_seed))
                        except (TypeError, ValueError):
                            seed_val = -1

                        sampler = (t2i_sampler or "Euler a").strip()
                        steps   = max(1, int(float(t2i_steps or 28)))
                        cfg     = float(t2i_cfg or 7.0)
                        width   = int(float(t2i_w or 1024))
                        height  = int(float(t2i_h or 1024))
                        neg     = (t2i_neg or "").strip()

                        _log(f"  → prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                        _log(f"  → sampler: {sampler} · steps: {steps} · cfg: {cfg} · {width}x{height} · seed: {seed_val}")

                        p = processing.StableDiffusionProcessingTxt2Img(
                            sd_model       = shared.sd_model,
                            outpath_samples= shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
                            outpath_grids  = shared.opts.outdir_grids   or shared.opts.outdir_txt2img_grids,
                            prompt         = prompt,
                            negative_prompt= neg,
                            sampler_name   = sampler,
                            steps          = steps,
                            cfg_scale      = cfg,
                            width          = width,
                            height         = height,
                            seed           = seed_val,
                            n_iter         = 1,
                            batch_size     = 1,
                        )

                        # Empty ScriptRunner: built-in scripts would crash on missing arguments.
                        try:
                            p.scripts = _scripts_mod.ScriptRunner()
                            p.script_args = ()
                        except Exception as _e:
                            _log(f"  → ScriptRunner setup skipped: {_e}")
                            p.scripts = None
                            p.script_args = ()

                        processed = None
                        with closing(p):
                            processed = processing.process_images(p)

                        _log(f"  → processed type: {type(processed).__name__}, "
                             f"images: {len(processed.images) if processed and hasattr(processed, 'images') else 'N/A'}")

                        if not processed or not processed.images:
                            return (f"✅  Merge saved to `{op}`\n⚠️  Generation produced no image.",
                                    1.0, gr.update(visible=False), None, "")

                        img      = processed.images[0]
                        gen_info = (f"Prompt: {prompt}\n"
                                    f"Negative: {neg}\n"
                                    f"Sampler: {sampler} · Steps: {steps} · CFG: {cfg} · "
                                    f"Size: {width}x{height} · Seed: {processed.seed}")

                        _log("Merge & Gen — done.")
                        return (f"✅  Merged + generated. Model loaded: `{os.path.basename(op)}`",
                                1.0,
                                gr.update(visible=True),
                                img,
                                gen_info)
                    except Exception as e:
                        import traceback
                        _log(f"Generation failed: {e}\n{traceback.format_exc()}")
                        return (f"✅  Merge saved to `{op}`\n❌  Generation failed: {e}",
                                1.0, gr.update(visible=False), None, "")

                def bm_sweep(method, base_alpha, mode, ma, mb, mc,
                             out_n, precision, vae, save_meta,
                             sweep_spec, sweep_keep,
                             t2i_prompt, t2i_neg, t2i_steps, t2i_cfg,
                             t2i_w, t2i_h, t2i_seed, t2i_sampler,
                             density, epsilon, gamma, use_ties_variant,
                             nuslerp_rowwise, anima_fill, anima_map, norm_clamp,
                             legacy_taskvec, *extra):
                    """Merge at several alphas, generate one image per alpha with a
                    fixed seed, and return a labeled comparison grid + gallery."""
                    _hide = gr.update(visible=False)
                    extra_mb, extra_w = _split_extra_donors(extra)
                    if not ma or not mb:
                        return "❌  Select both Model A and Model B.", 0.0, _hide, None, None
                    if method in METHODS_REQUIRING_C and not mc:
                        return f"❌  {method} requires Model C.", 0.0, _hide, None, None

                    try:
                        alphas = _parse_alpha_sweep(sweep_spec)
                    except Exception as e:
                        return f"❌  Alpha spec: {e}", 0.0, _hide, None, None

                    pa = find_model(ma)
                    pb = find_model(mb)
                    pc = find_model(mc) if mc else None
                    vp = find_vae(vae) if vae and vae != "— keep original —" else None
                    if _opt(OPT_AUTO_NAME, AUTO_NAME_EMPTY) == AUTO_NAME_ALWAYS \
                            or not _clean_filename(out_n):
                        base_name = auto_merge_name(method, None, ma, mb, mc, extra_mb,
                                                    suffix="sweep")
                    else:
                        base_name = _clean_filename(out_n)

                    # Fixed generation settings — resolve the seed ONCE so every
                    # alpha is compared on identical noise.
                    prompt = (t2i_prompt or "").strip() or "masterpiece, best quality, 1girl, portrait"
                    try:
                        seed_val = int(float(t2i_seed))
                    except (TypeError, ValueError):
                        seed_val = -1
                    if seed_val < 0:
                        import random as _random
                        seed_val = _random.randint(0, 2**31 - 1)
                        _log(f"Alpha Sweep — random seed fixed at {seed_val} for all steps.")
                    sampler = (t2i_sampler or "Euler a").strip()
                    steps   = max(1, int(float(t2i_steps or 28)))
                    cfg     = float(t2i_cfg or 7.0)
                    width   = int(float(t2i_w or 1024))
                    height  = int(float(t2i_h or 1024))
                    neg     = (t2i_neg or "").strip()

                    _log(f"Alpha Sweep — {len(alphas)} steps: {alphas}")
                    _log(f"  → sampler: {sampler} · steps: {steps} · cfg: {cfg} · "
                         f"{width}x{height} · seed: {seed_val}")

                    labeled  = []
                    gallery  = []
                    files    = []
                    stopped  = False
                    errors   = []

                    for i, a in enumerate(alphas):
                        _log(f"Alpha Sweep — step {i + 1}/{len(alphas)} · α = {a}")
                        if mode in ("Anima", "Anima Easy"):
                            bw = [float(a)] * _anima_ui_depth(ma)
                        else:
                            bw = [float(a)] * 20
                            bw[0] = float(base_alpha)

                        op = resolve_output_path(
                            get_ckpt_dir(), f"{base_name}_a{a:.2f}", f"{base_name}_a{a:.2f}",
                            inputs=[pa, pb, pc, *extra_mb])
                        meta = None
                        if save_meta:
                            meta = {
                                "neomerger_version": NEOMERGER_VERSION,
                                "merge_type":        "alpha_sweep",
                                "merge_method":      method,
                                "model_a":           os.path.basename(pa),
                                "model_b":           os.path.basename(pb),
                                "model_c":           os.path.basename(pc) if pc else "N/A",
                                "alpha":             str(a),
                                "base_alpha":        str(round(float(base_alpha), 4)),
                                "sweep":             json.dumps(alphas),
                                "precision":         precision,
                                "vae":               os.path.basename(vp) if vp else "original",
                            }

                        try:
                            r = merge_models_method(
                                pa, [pb] + extra_mb, pc, bw, float(a),
                                method, op, precision, vp, meta,
                                density=float(density), epsilon=float(epsilon),
                                gamma=float(gamma),
                                nuslerp_row_wise=bool(nuslerp_rowwise),
                                use_ties_variant=bool(use_ties_variant),
                                anima_unmapped_fill=anima_fill,
                                anima_map_source=anima_map,
                                norm_clamp=float(norm_clamp),
                                donor_weights=[1.0] + extra_w,
                                legacy_taskvec=bool(legacy_taskvec),
                            )
                            if r is False:
                                stopped = True
                                break
                        except Exception as e:
                            errors.append(f"α={a}: merge failed — {e}")
                            continue
                        files.append(op)

                        err = None
                        try:
                            err = _nm_load_checkpoint_into_ui(op)
                        except Exception as e:
                            err = str(e)
                        if err:
                            errors.append(f"α={a}: load failed — {err}")
                            continue

                        try:
                            img = _nm_generate_preview(prompt, neg, steps, cfg,
                                                       width, height, seed_val, sampler)
                        except Exception as e:
                            errors.append(f"α={a}: generation failed — {e}")
                            continue

                        lab = _nm_label_image(img, f"α = {a:g}")
                        labeled.append(lab)
                        gallery.append((lab, f"alpha {a:g}"))

                        if _STOP_MERGE[0]:
                            stopped = True
                            break

                    if not labeled:
                        msg = "❌  Sweep produced no images."
                        if errors:
                            msg += "\n" + "\n".join(f"- {e}" for e in errors)
                        return msg, 0.0, _hide, None, None

                    grid = _nm_make_grid(labeled)

                    # Cleanup: keep the LAST merged file (it is the currently
                    # loaded model — deleting it would leave the UI pointing at
                    # a missing checkpoint), delete the rest unless keeping all.
                    cleanup_note = ""
                    if not sweep_keep and files:
                        keep_file = files[-1]
                        deleted, failed = 0, 0
                        for f in files[:-1]:
                            try:
                                os.remove(f)
                                deleted += 1
                            except Exception:
                                failed += 1
                        sd_models.list_models()
                        cleanup_note = (f"\n🧹  Deleted {deleted} intermediate models · "
                                        f"kept `{os.path.basename(keep_file)}` (currently loaded)")
                        if failed:
                            cleanup_note += f" · ⚠️ {failed} could not be deleted (file in use)"
                    elif sweep_keep and files:
                        cleanup_note = f"\n💾  Kept {len(files)} models in the checkpoints folder"

                    status = (f"{'⏹  Sweep stopped early' if stopped else '✅  Sweep complete'} — "
                              f"{len(labeled)}/{len(alphas)} alphas rendered · seed {seed_val}"
                              f"{cleanup_note}")
                    if errors:
                        status += "\n" + "\n".join(f"⚠️ {e}" for e in errors)

                    return status, 1.0, gr.update(visible=True), grid, gallery

                def bm_save_p(name, method, alpha, base_alpha, use_blocks, mode, use_exp, *vals):
                    if not name.strip(): return "❌  Enter a name.", gr.update()
                    ne = len(bm_easy_sliders)
                    nn = len(bm_block_sliders)
                    na = len(bm_anima_sliders)
                    save_preset(name.strip(), {
                        "method": method, "alpha": alpha,
                        "base_alpha": base_alpha,
                        "use_blocks": use_blocks, "mode": mode,
                        "use_exp": bool(use_exp),
                        "easy":   dict(zip(bm_easy_sliders.keys(),  vals[:ne])),
                        "normal": dict(zip(bm_block_sliders.keys(), vals[ne:ne+nn])),
                        "anima":  dict(zip(bm_anima_sliders.keys(), vals[ne+nn:ne+nn+na])),
                        "anima_easy": dict(zip(bm_anima_easy_sliders.keys(),
                                               vals[ne+nn+na:])),
                    }, "bm_")
                    return f"✅  Saved **{name}**.", gr.update(choices=list_presets("bm_"))

                def bm_load_p(name):
                    if not name:
                        return ["❌  Select a preset first."] + [gr.update()] * (len(bm_all) + 6)
                    d = load_preset(name, "bm_")
                    if not d:
                        return [f"❌  Preset '{name}' not found. Try saving it again."] + [gr.update()] * (len(bm_all) + 6)

                    easy_data  = d.get("easy", {})
                    norm_data  = d.get("normal", {})
                    anima_data = d.get("anima", {})

                    ev = []
                    for k in bm_easy_sliders.keys():
                        v = easy_data.get(k, 0.0)
                        try: ev.append(float(v))
                        except (TypeError, ValueError): ev.append(0.0)

                    nv = []
                    for k in bm_block_sliders.keys():
                        v = norm_data.get(k, 0.5)
                        try: nv.append(float(v))
                        except (TypeError, ValueError): nv.append(0.5)

                    av = []
                    for k in bm_anima_sliders.keys():
                        v = anima_data.get(k, 0.5)
                        try: av.append(float(v))
                        except (TypeError, ValueError): av.append(0.5)

                    aev = []
                    ae_data = d.get("anima_easy", {})
                    for k in bm_anima_easy_sliders.keys():
                        v = ae_data.get(k, 0.0)
                        try: aev.append(float(v))
                        except (TypeError, ValueError): aev.append(0.0)

                    return ([f"✅  Loaded **{name}**.",
                             gr.update(value=d.get("method", "Weighted Sum")),
                             gr.update(value=float(d.get("alpha", 0.5))),
                             gr.update(value=float(d.get("base_alpha", 0.5))),
                             gr.update(value=bool(d.get("use_blocks", False))),
                             gr.update(value=d.get("mode", "Easy")),
                             gr.update(value=bool(d.get("use_exp", False)))]
                            + [gr.update(value=v) for v in ev + nv + av + aev])

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
                        # Lazy open, as in the merge engine.
                        sd_a = open_model(pa, lazy=True)
                        sd_b = open_model(pb, lazy=True)
                    except Exception as e:
                        return f"❌  Could not load models: {e}", ""

                    # DiT models are binned by their own blocks, not the SDXL map.
                    arch = detect_arch(pa)
                    if arch in DIT_ARCHS:
                        block_index_fn = block_index_fn_for(arch)
                        block_labels   = anima_block_labels(anima_block_count(pa))
                        semantic_for   = lambda i: []          # no SDXL semantics for DiT
                    else:
                        block_index_fn = get_ckpt_block_index
                        block_labels   = CKPT_BLOCK_LABELS
                        semantic_for   = lambda i: BLOCK_SEMANTIC_LABELS.get(i, [])

                    # Compute relative difference per block
                    # diff = mean(|A-B| / (|A|+|B|+eps) * 2)
                    # 0% = identical, 100% = completely different
                    block_diffs  = {}
                    block_counts = {}

                    # Match keys by normalised tail when A and B use different prefixes.
                    def _prefix_of(klist):
                        for k in klist:
                            nn = _normalise_key(k)
                            if nn != k:
                                return k[:len(k) - len(nn)]
                        return ""
                    a_pref = _prefix_of(list(sd_a.keys()))
                    b_pref = _prefix_of(list(sd_b.keys()))
                    b_lookup = None
                    if a_pref != b_pref:
                        b_lookup = _build_normalised_lookup(list(sd_b.keys()))

                    skipped_shape   = 0
                    normalised_used = 0
                    for key in sd_a.keys():
                        # Find B's counterpart, allowing a prefix difference.
                        if key in sd_b:
                            bkey = key
                        elif b_lookup is not None:
                            bkey = b_lookup.get(_normalise_key(key))
                            if bkey is not None:
                                normalised_used += 1
                        else:
                            bkey = None
                        if bkey is None:
                            continue
                        t_a = sd_a[key]
                        t_b = sd_b[bkey]
                        # Skip if shapes differ — incompatible architectures
                        if t_a.shape != t_b.shape:
                            skipped_shape += 1
                            continue
                        idx   = block_index_fn(key)
                        t_af  = t_a.float()
                        t_bf  = t_b.float()
                        diff  = (t_af - t_bf).abs()
                        denom = (t_af.abs() + t_bf.abs() + 1e-8)
                        rel   = (diff / denom * 2).mean().item()
                        rel   = max(0.0, min(1.0, rel))
                        block_diffs[idx]  = block_diffs.get(idx, 0.0) + rel
                        block_counts[idx] = block_counts.get(idx, 0) + 1

                    close_model(sd_a); close_model(sd_b)
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
                    for i, label in enumerate(block_labels):
                        pct      = results.get(i, 0.0)
                        col      = bar_color(pct)
                        bar_w    = min(100, pct * 2)  # scale for visibility
                        sem_tags = semantic_for(i)
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
                    most_diff_lbl = block_labels[most_diff_idx] if most_diff_idx < len(block_labels) else "?"
                    most_diff_sem_tags = semantic_for(most_diff_idx)
                    most_diff_sem = ", ".join(most_diff_sem_tags) if most_diff_sem_tags else "—"

                    norm_note = ""
                    if normalised_used > 0:
                        norm_note = (
                            f'<div style="padding:8px 12px;margin-bottom:12px;'
                            f'background:var(--color-background-secondary);border-radius:6px;'
                            f'font-size:12px;border-left:3px solid rgb(240,190,50)">'
                            f'ℹ️ <strong>Keys normalised:</strong> Model A and Model B use '
                            f'different key prefixes (e.g. <code>model.diffusion_model.</code> '
                            f'vs <code>net.</code>). {normalised_used} keys were matched by their '
                            f'normalised name so the comparison is valid. The same normalisation '
                            f'is applied automatically when you merge these two.'
                            f'</div>'
                        )

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
                        f'{norm_note}{summary}{legend}'
                        f'<table style="width:100%;border-collapse:collapse">{rows}</table>'
                        f'</div>'
                    )

                    warn = ""
                    if skipped_shape > 0:
                        warn = f" ⚠️ {skipped_shape} keys skipped (shape mismatch — different architectures)"
                    norm_status = ""
                    if normalised_used > 0:
                        norm_status = f" · ℹ️ {normalised_used} keys matched via prefix normalisation"
                    return f"✅  Analysis complete — {len(results)} blocks.{warn}{norm_status}", html

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
                    inputs=[bm_method, bm_alpha, bm_base_alpha, bm_use_blocks, bm_mode,
                             bm_model_a, bm_model_b, bm_model_c,
                             bm_out_name, bm_precision, bm_vae, bm_save_meta, bm_use_exp,
                             bm_density, bm_epsilon, bm_gamma, bm_use_ties_variant,
                             bm_nuslerp_rowwise, bm_allow_negative, bm_anima_fill, bm_anima_map,
                             bm_norm_clamp, bm_legacy_taskvec]
                           + bm_all + bm_extra_models + bm_extra_weights,
                    outputs=[bm_status, bm_progress]
                )
                bm_mergegen_btn.click(
                    fn=bm_mergegen,
                    inputs=[bm_method, bm_alpha, bm_base_alpha, bm_use_blocks, bm_mode,
                            bm_model_a, bm_model_b, bm_model_c,
                            bm_out_name, bm_precision, bm_vae, bm_save_meta, bm_use_exp,
                            bm_t2i_prompt, bm_t2i_neg, bm_t2i_steps, bm_t2i_cfg,
                            bm_t2i_w, bm_t2i_h, bm_t2i_seed, bm_t2i_sampler,
                            bm_density, bm_epsilon, bm_gamma, bm_use_ties_variant,
                            bm_nuslerp_rowwise, bm_allow_negative, bm_anima_fill, bm_anima_map,
                            bm_norm_clamp, bm_legacy_taskvec]
                           + bm_all + bm_extra_models + bm_extra_weights,
                    outputs=[bm_status, bm_progress, bm_gen_panel, bm_gen_image, bm_gen_info]
                )
                bm_sweep_btn.click(
                    fn=bm_sweep,
                    inputs=[bm_method, bm_base_alpha, bm_mode,
                            bm_model_a, bm_model_b, bm_model_c,
                            bm_out_name, bm_precision, bm_vae, bm_save_meta,
                            bm_sweep_spec, bm_sweep_keep,
                            bm_t2i_prompt, bm_t2i_neg, bm_t2i_steps, bm_t2i_cfg,
                            bm_t2i_w, bm_t2i_h, bm_t2i_seed, bm_t2i_sampler,
                            bm_density, bm_epsilon, bm_gamma, bm_use_ties_variant,
                            bm_nuslerp_rowwise, bm_anima_fill, bm_anima_map, bm_norm_clamp, bm_legacy_taskvec]
                           + bm_extra_models + bm_extra_weights,
                    outputs=[bm_status, bm_progress, bm_sweep_panel,
                             bm_sweep_grid, bm_sweep_gallery]
                )
                bm_p_save.click(bm_save_p,
                    inputs=[bm_p_name, bm_method, bm_alpha, bm_base_alpha, bm_use_blocks, bm_mode, bm_use_exp] + bm_all,
                    outputs=[bm_p_status, bm_p_dd])
                bm_p_load.click(bm_load_p,
                    inputs=[bm_p_dd],
                    outputs=[bm_p_status, bm_method, bm_alpha, bm_base_alpha, bm_use_blocks, bm_mode, bm_use_exp] + bm_all)
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
                                    bi_ref     = gr.Dropdown(
                                        [""] + ml, value="",
                                        label="LoRA reference model  (optional)",
                                        info="Only needed when the LoRA was trained on a "
                                             "different DiT depth than the checkpoint "
                                             "(e.g. a 28-block Anima LoRA onto a 40-block "
                                             "upscale). Pick the model the LoRA was trained on."
                                    )
                                    bi_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                                    bi_refresh.click(
                                        fn=lambda: [gr.update(choices=get_model_list()),
                                                    gr.update(choices=get_lora_list()),
                                                    gr.update(choices=[""] + get_model_list())],
                                        outputs=[bi_ckpt, bi_lora, bi_ref]
                                    )

                                gr.Markdown("")

                                with gr.Group():
                                    gr.Markdown("### 💾 Output")
                                    bi_out_name = gr.Textbox(label="File name", value="",
                                                             placeholder=_NAME_PLACEHOLDER)
                                    gr.Markdown("<small>Saved as `.safetensors` in your checkpoints folder. Existing files are kept: a new one gets `_2`, `_3`, … (change in Settings → NeoMerger).</small>")

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

                        def bi_run(mode, ckpt, lora, out_n, precision, vae, save_meta,
                                   lora_ref, strength, *adv_vals):
                            if not ckpt or not lora:
                                return "❌  Select a Checkpoint and a LoRA.", 0.0
                            strength  = float(strength) if strength is not None else 1.0
                            adv_vals  = tuple(
                                float(v) if v is not None and not isinstance(v, str)
                                or (isinstance(v, str) and v.replace('.','',1).lstrip('-').isdigit())
                                else 0.0
                                for v in adv_vals
                            )
                            pc  = find_model(ckpt)
                            # Architecture first: DiT models need one weight per block.
                            try:
                                _bi_arch = detect_arch(pc)
                            except Exception:
                                _bi_arch = "sdxl"
                            _n_blocks = (anima_block_count(pc)
                                         if _bi_arch in DIT_ARCHS else 20)

                            if mode == "Easy":
                                bw = [strength] * _n_blocks
                            elif _bi_arch in DIT_ARCHS:
                                # The Advanced semantic sliders are mapped onto SDXL
                                # UNet blocks and don't translate to DiT blocks —
                                # fall back to a flat Strength.
                                _log("⚠  Advanced semantic sliders are SDXL-only — "
                                     f"using flat Strength for this "
                                     f"{DIT_ARCH_NAMES[_bi_arch]} checkpoint.")
                                bw = [strength] * _n_blocks
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
                            pl  = find_lora(lora)
                            vp  = find_vae(vae) if vae and vae != "— keep original —" else None
                            _strength_tag = f"s{_num(strength)}" if mode == "Easy" else "adv"
                            op  = resolve_output_path(
                                get_ckpt_dir(), out_n,
                                f"{_name_stem(ckpt)}+{_name_stem(lora)}_{_strength_tag}",
                                inputs=[pc, pl])
                            try:
                                fused = bake_lora_into_checkpoint(pc, pl, bw, op,
                                                                   precision=precision,
                                                                   vae_path=vp,
                                                                   save_metadata=save_meta,
                                                                   lora_ref_model=(lora_ref or None))
                                _r = _LAST_BAKE_REPORT[0] or {}
                                _tot = _r.get("modules_parsed", fused)
                                _unp = sum(len(v) for v in (_r.get("unparsed") or {}).values())
                                if _tot and (fused < _tot or _unp):
                                    return (f"⚠️  Baked {fused}/{_tot} LoRA modules"
                                            + (f", {_unp} tensors in an unsupported format" if _unp else "")
                                            + f" → `{op}` (see terminal for the match report)"), 1.0
                                return f"✅  Baked {fused} layers → `{op}`", 1.0
                            except Exception as e:
                                import traceback
                                _log(f"Bake-in error: {traceback.format_exc()}")
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
                                     bi_precision, bi_vae, bi_save_meta, bi_ref] + bi_all,
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
                                    lm_out_name = gr.Textbox(label="File name", value="",
                                                             placeholder=_NAME_PLACEHOLDER)
                                    gr.Markdown("<small>Saved as `.safetensors` in your LoRA folder. Existing files are kept: a new one gets `_2`, `_3`, … (change in Settings → NeoMerger).</small>")

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
                            op  = resolve_output_path(
                                get_lora_dir(), out_n,
                                f"{_name_stem(la)}+{_name_stem(lb)}_w{_num(weight)}",
                                inputs=[pa, pb])
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

                    # ─────────────────────────────────────────────────────
                    #  SUB-TAB — EXTRACT  (Checkpoint − Checkpoint → LoRA)
                    # ─────────────────────────────────────────────────────
                    with gr.Tab("⛏️  Extract  (Checkpoint − Checkpoint → LoRA)"):

                        gr.Markdown(
                            "Extract a LoRA from the **difference between two checkpoints** "
                            "(SVD low-rank approximation): `base + LoRA ≈ finetuned`. "
                            "Works on SDXL/Illustrious/Pony and **Anima**. "
                            "Output is kohya-format, directly usable in Forge and in the Bake-in tab."
                        )
                        gr.Markdown("")

                        with gr.Row(equal_height=True):

                            with gr.Column(scale=1, min_width=300):
                                with gr.Group():
                                    gr.Markdown("### 📂 Models")
                                    ml3 = get_model_list()
                                    ex_model_ft   = gr.Dropdown(ml3, label="Finetuned / merged model  (A)")
                                    ex_model_base = gr.Dropdown(ml3, label="Base model  (B)")
                                    ex_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
                                    ex_refresh.click(
                                        fn=lambda: [gr.update(choices=get_model_list())] * 2,
                                        outputs=[ex_model_ft, ex_model_base]
                                    )

                                gr.Markdown("")

                                with gr.Group():
                                    gr.Markdown("### 💾 Output")
                                    ex_out_name = gr.Textbox(label="File name", value="",
                                                             placeholder=_NAME_PLACEHOLDER)
                                    gr.Markdown("<small>Saved as `.safetensors` in your LoRA folder. Existing files are kept: a new one gets `_2`, `_3`, … (change in Settings → NeoMerger).</small>")

                                gr.Markdown("")

                                with gr.Accordion("⚙️ Advanced Options", open=False):
                                    ex_precision = gr.Radio(
                                        ["fp16", "bf16", "fp32"], value="fp16",
                                        label="Output precision"
                                    )
                                    ex_clamp = gr.Slider(
                                        0.90, 1.0, value=0.99, step=0.005,
                                        label="Outlier clamp (quantile)",
                                        info="Clamps extreme SVD values. 0.99 = kohya default · 1.0 = no clamp."
                                    )
                                    ex_gpu = gr.Checkbox(
                                        value=True, label="Use GPU for SVD",
                                        info="Much faster. Falls back to CPU per-tensor on OOM."
                                    )
                                    ex_save_meta = gr.Checkbox(value=True, label="Save extraction metadata")

                                gr.Markdown("")
                                with gr.Row():
                                    ex_run_btn  = gr.Button("⛏️  Extract LoRA", variant="primary", scale=3)
                                    ex_stop_btn = gr.Button("⏹  Stop",           variant="stop",    scale=1)
                                ex_status   = gr.Markdown("")
                                ex_progress = gr.Slider(0.0, 1.0, value=0.0, label="Progress",
                                                        interactive=False)

                            with gr.Column(scale=2):
                                with gr.Group():
                                    gr.Markdown("### 🎚️ Rank")
                                    ex_rank = gr.Slider(
                                        1, 256, value=32, step=1,
                                        label="LoRA rank (dim)",
                                        info="Higher = more faithful to the difference, bigger file. "
                                             "32–64 covers most style/character diffs · 128+ for heavy finetunes."
                                    )
                                    ex_conv_rank = gr.Slider(
                                        0, 128, value=0, step=1,
                                        label="Conv rank (LoCon) — SDXL only",
                                        info="Also extract 3x3 conv layers (LoCon-style). 0 = off. "
                                             "Try 16–32 if the diff includes texture/detail changes."
                                    )
                                    ex_include_te = gr.Checkbox(
                                        value=True, label="Include text encoders (SDXL)",
                                        info="Extract CLIP-L and OpenCLIP-G differences too. "
                                             "Ignored for Anima (Qwen3 TE is a separate file)."
                                    )
                                gr.Markdown("")
                                gr.Markdown(
                                    "### ℹ️ Notes\n"
                                    "- **A − B**: A is the model containing what you want to capture, "
                                    "B is what you want to subtract (usually the base A was trained/merged from).\n"
                                    "- The extracted LoRA at **strength 1.0** approximates A when applied to B.\n"
                                    "- Identical layers are skipped automatically — extracting a model "
                                    "from itself produces an error, not an empty LoRA.\n"
                                    "- **Anima**: extraction targets the DiT blocks (`blocks.*`), matching "
                                    "what `networks.lora_anima` trains. Keys round-trip cleanly through "
                                    "the Bake-in tab.\n"
                                    "- Verify the result with **Bake-in** onto the base model and compare "
                                    "against the original with **Block Similarity**."
                                )

                        def ex_run(ma, mb, rank, conv_rank, clamp_q, include_te,
                                   out_n, precision, use_gpu, save_meta):
                            if not ma or not mb:
                                return "❌  Select both models.", 0.0
                            if ma == mb:
                                return "❌  Finetuned and base model are the same file.", 0.0
                            pa = find_model(ma)
                            pb = find_model(mb)
                            op = resolve_output_path(
                                get_lora_dir(), out_n,
                                f"{_name_stem(ma)}-minus-{_name_stem(mb)}_r{int(rank)}",
                                inputs=[pa, pb])
                            try:
                                n = extract_lora_from_models(
                                    pa, pb, op,
                                    rank=int(rank), conv_rank=int(conv_rank),
                                    clamp_q=float(clamp_q),
                                    include_te=bool(include_te),
                                    precision=precision,
                                    save_metadata=bool(save_meta),
                                    use_gpu=bool(use_gpu),
                                )
                                if n is None:
                                    return "⏹  Extraction stopped by user.", 0.0
                                return f"✅  Extracted {n} modules → `{op}`", 1.0
                            except Exception as e:
                                import traceback
                                _log(f"Extraction failed: {e}\n{traceback.format_exc()}")
                                return f"❌  {e}", 0.0

                        def ex_stop():
                            _STOP_MERGE[0] = True
                            _log("⏹  Stop requested by user...")
                            return "⏹  Stop requested — finishing current operation..."

                        ex_stop_btn.click(fn=ex_stop, outputs=[ex_status])

                        ex_run_btn.click(ex_run,
                            inputs=[ex_model_ft, ex_model_base, ex_rank, ex_conv_rank,
                                    ex_clamp, ex_include_te, ex_out_name,
                                    ex_precision, ex_gpu, ex_save_meta],
                            outputs=[ex_status, ex_progress])

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
                            # dtype/shape from the header; no tensor is read.
                            _dtype_map = {
                                "F64": "float64", "F32": "float32", "F16": "float16",
                                "BF16": "bfloat16", "I64": "int64", "I32": "int32",
                                "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
                            }
                            with safe_open(path, framework="pt", device="cpu") as sf:
                                raw_meta   = dict(sf.metadata() or {})
                                for k in sf.keys():
                                    total_keys += 1
                                    sl  = sf.get_slice(k)
                                    raw = sl.get_dtype()
                                    dt  = _dtype_map.get(raw, raw)
                                    key_dtypes[dt] = key_dtypes.get(dt, 0) + 1
                                    n = 1
                                    for s in sl.get_shape():
                                        n *= s
                                    total_params += n
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

                    # DiT architectures first, through the shared detect_arch.
                    try:
                        _dit = detect_arch(path)
                    except Exception:
                        _dit = "sdxl"

                    if _dit in DIT_ARCHS:
                        arch = f"{DIT_ARCH_NAMES[_dit]} (DiT)"

                    # Flux variants
                    elif has(["double_blocks","single_blocks","transformer_blocks.0.attn.norm"], sample_keys):
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

            # ═════════════════════════════════════════════════════════════
            #  TAB — BLOCK PROBE
            # ═════════════════════════════════════════════════════════════
            with gr.Tab("🧪  Block Probe"):
                gr.Markdown("### 🧪 Block Probe")
                gr.Markdown(
                    "<small>Swaps one block at a time with **Model B** weights "
                    "(in memory — no disk merge) and generates one image per block. "
                    "Uses the current **txt2img** settings (prompt, steps, CFG, seed…). "
                    "Set a fixed seed in txt2img for comparable cells.</small>"
                )
                gr.Markdown("---")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        with gr.Group():
                            gr.Markdown("### 📂 Model B")
                            pr_model_b = gr.Dropdown(get_model_list(), label="Model B  (block source)")
                            pr_refresh = gr.Button("🔄  Refresh", size="sm")
                            pr_arch = gr.Radio(["sdxl", "anima", "krea2", "moss"], value="sdxl", label="Architecture")

                        with gr.Group():
                            gr.Markdown("### 🎚️ Mode")
                            pr_mode = gr.Radio(
                                ["single", "easy", "all"],
                                value="single",
                                label="Probe mode",
                                info="single = one block per image · "
                                     "easy = one image per semantic group · "
                                     "all = selected blocks together in one image",
                            )

                        # --- block selector (single / all modes) ---
                        with gr.Group(visible=True) as pr_blocks_panel:
                            gr.Markdown("### 🧱 Blocks")
                            pr_blocks = gr.Dropdown(
                                list(PROBE_SDXL_BLOCKS.keys()),
                                label="Blocks  (empty = all, in 'single' mode)",
                                multiselect=True, value=[],
                            )

                        # --- semantic group selector (easy mode) ---
                        with gr.Group(visible=False) as pr_easy_panel:
                            gr.Markdown("### 🎨 Semantic groups")
                            _easy_choices = [c for c in EASY_CATEGORIES if not c.startswith("⚗️")]
                            pr_easy = gr.Dropdown(
                                _easy_choices,
                                label="Groups to test  (empty = all)",
                                multiselect=True, value=[],
                            )

                        pr_alpha = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.05,
                            label="Blend toward B  (1 = fully from B)"
                        )

                        pr_run = gr.Button("🧪  Run Probe", variant="primary",
                                           elem_id="neomerger_probe_btn")
                        pr_status = gr.Textbox(label="Status", interactive=False, lines=2)

                        # Hidden fields — populated by neomerger.js (prefix neomerger_pr)
                        pr_t2i_prompt  = gr.Textbox(visible=False, elem_id="neomerger_pr_prompt")
                        pr_t2i_neg     = gr.Textbox(visible=False, elem_id="neomerger_pr_neg")
                        pr_t2i_steps   = gr.Number(visible=False, value=28, elem_id="neomerger_pr_steps")
                        pr_t2i_cfg     = gr.Number(visible=False, value=7.0, elem_id="neomerger_pr_cfg")
                        pr_t2i_w       = gr.Number(visible=False, value=1024, elem_id="neomerger_pr_w")
                        pr_t2i_h       = gr.Number(visible=False, value=1024, elem_id="neomerger_pr_h")
                        pr_t2i_seed    = gr.Number(visible=False, value=-1, elem_id="neomerger_pr_seed")
                        pr_t2i_sampler = gr.Textbox(visible=False, value="Euler a", elem_id="neomerger_pr_sampler")

                    with gr.Column(scale=2):
                        pr_gallery = gr.Gallery(label="Probe results", columns=4,
                                                height="auto", object_fit="contain")

                # show the right selector for the chosen mode
                def _pr_mode_change(m):
                    is_easy = (m == "easy")
                    return (gr.update(visible=not is_easy),   # blocks panel
                            gr.update(visible=is_easy))       # easy panel
                pr_mode.change(_pr_mode_change, inputs=[pr_mode],
                               outputs=[pr_blocks_panel, pr_easy_panel])

                def _pr_arch_change(a, model_b):
                    # Keep the real depth when the user flips the arch dropdown
                    # by hand, otherwise the list collapses back to 28 entries
                    # and the deepest blocks of an upscale disappear.
                    n = None
                    if is_dit_arch(a) and model_b:
                        try:
                            n = anima_block_count(find_model(model_b))
                        except Exception:
                            n = None
                    return gr.update(choices=list(probe_block_map(a, n).keys()), value=[])
                pr_arch.change(_pr_arch_change, inputs=[pr_arch, pr_model_b], outputs=[pr_blocks])

                # Auto-detect architecture when Model B is chosen, so the user
                # doesn't have to know whether it's SDXL or Anima. detect_arch
                # already handles all the Anima save-format variants. The radio
                # stays editable as a manual override if detection is ever wrong.
                def _pr_model_change(model_b):
                    if not model_b:
                        return gr.update(), gr.update()
                    n = None
                    try:
                        p = find_model(model_b)
                        a = detect_arch(p)
                        if a in DIT_ARCHS:
                            n = anima_block_count(p)     # depth-upscaled models
                    except Exception:
                        a = "sdxl"
                    return (gr.update(value=a),
                            gr.update(choices=list(probe_block_map(a, n).keys()), value=[]))
                pr_model_b.change(_pr_model_change, inputs=[pr_model_b],
                                  outputs=[pr_arch, pr_blocks])

                pr_refresh.click(lambda: gr.update(choices=get_model_list()),
                                 outputs=[pr_model_b])

                pr_run.click(
                    fn=run_block_probe,
                    inputs=[pr_model_b, pr_arch, pr_mode, pr_blocks, pr_easy, pr_alpha,
                            pr_t2i_prompt, pr_t2i_neg, pr_t2i_steps, pr_t2i_cfg,
                            pr_t2i_w, pr_t2i_h, pr_t2i_seed, pr_t2i_sampler],
                    outputs=[pr_status, pr_gallery],
                )


    return [(ui, "NeoMerger", "neo_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)
