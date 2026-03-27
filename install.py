"""
install.py — NeoMerger
Installa safetensors se non presente (di solito già incluso in Forge).
"""
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import safetensors
except ImportError:
    print("[NeoMerger] Installazione safetensors...")
    pip("safetensors")

print("[NeoMerger] Dipendenze OK.")
