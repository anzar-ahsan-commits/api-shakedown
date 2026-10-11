"""Writes the generated test harness to disk."""

from pathlib import Path

from api_shakedown.agents.coordinator import Harness


class HarnessExistsError(Exception):
    pass


def write_harness(harness: Harness, out_dir: Path, *, force: bool = False) -> list[Path]:
    """Write every harness file under `out_dir`.

    Refuses a non-empty directory unless `force`; even then only the harness's own files
    are overwritten and nothing else is deleted.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise HarnessExistsError(f"{out_dir} is not empty; use --force to overwrite.")
    written = []
    for relative, content in sorted(harness.files.items()):
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return written
