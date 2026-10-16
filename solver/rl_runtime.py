"""Expose the bundled solver package to the isolated runtime."""
from pathlib import Path
import sys


def bootstrap():
    root=Path(__file__).resolve().parent
    source=str(root/'src')
    if source not in sys.path:sys.path.insert(0,source)
    return root
