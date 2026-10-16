"""Small subprocess boundary for a native local folder chooser."""

from __future__ import annotations

import sys
from pathlib import Path


def choose_folder(initial: str = "") -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        initialdir=str(Path(initial).expanduser()) if initial else None,
        title="WAAM Validator 입력 폴더 선택",
        mustexist=True,
    )
    root.destroy()
    return str(selected)


def main() -> int:
    selected = choose_folder(sys.argv[1] if len(sys.argv) > 1 else "")
    if selected:
        print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
