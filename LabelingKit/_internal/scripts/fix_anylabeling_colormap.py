"""
Hotfix for AnyLabeling startup error:
ValueError: assignment destination is read-only

This patches label_widget.py to ensure LABEL_COLORMAP is writable.
"""

import re
import site
from pathlib import Path


def patch_file(target: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, f"target not found: {target}"

    content = target.read_text(encoding="utf-8")

    # Already patched (specific LABEL_COLORMAP line)
    already = re.search(
        r"^\s*LABEL_COLORMAP\s*=\s*imgviz\.label_colormap\([^)]*\)\.copy\(\)",
        content,
        flags=re.MULTILINE,
    )
    if already:
        return False, "already patched"

    pattern = r"(^\s*LABEL_COLORMAP\s*=\s*imgviz\.label_colormap\([^)]*\))(?!\.copy\(\))"
    replacement = r"\1.copy()"
    patched, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        return False, "pattern not found"

    target.write_text(patched, encoding="utf-8")
    return True, "patched"


def find_target() -> Path | None:
    candidates: list[Path] = []
    for base in site.getsitepackages():
        candidates.append(Path(base) / "anylabeling" / "views" / "labeling" / "label_widget.py")

    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "anylabeling" / "views" / "labeling" / "label_widget.py")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def main() -> int:
    target = find_target()
    if target is None:
        print("[HOTFIX] target file not found (anylabeling not installed yet)")
        return 0

    changed, msg = patch_file(target)
    print(f"[HOTFIX] {msg}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
