"""Export the editable manuscript SVGs to vector PDF and 600-dpi PNG.

    python make_diagrams.py
    python make_diagrams.py --inkscape /path/to/inkscape

Requires Inkscape. Native export preserves mathematical subscripts, arrows,
physical page dimensions and embedded PDF fonts. SVGs remain the source of
truth: editing an SVG does not require rerunning either build_* generator.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
SVGS = ("architecture.svg", "graphical_abstract.svg")


def find_inkscape(explicit=None):
    if explicit:
        return explicit
    found = shutil.which("inkscape.com") or shutil.which("inkscape")
    if found:
        return found
    candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Inkscape/bin/inkscape.com",
        Path("/Applications/Inkscape.app/Contents/MacOS/inkscape"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit("Inkscape was not found. Supply its executable with --inkscape.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inkscape", help="Path to the Inkscape executable")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    args = parser.parse_args()
    inkscape = find_inkscape(args.inkscape)
    for name in SVGS:
        source = HERE / name
        for extension in ("pdf", "png"):
            destination = source.with_suffix("." + extension)
            command = [inkscape, str(source), "--export-area-page",
                       f"--export-type={extension}", f"--export-filename={destination}"]
            if extension == "png":
                command += [f"--export-dpi={args.dpi}", "--export-background=white"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode or not destination.is_file():
                raise RuntimeError(f"Export failed for {source.name}: {result.stderr}")
            print(f"{source.name} -> {destination.name} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
