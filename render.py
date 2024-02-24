import os
import sys
from inspect import getmembers
from pathlib import Path

from pynopegl_utils.export import export_workers
from pynopegl_utils.misc import SceneCfg
from pynopegl_utils.module import load_script

_OUTDIR = Path(__file__).resolve().parent / "output"


def _main():
    _OUTDIR.mkdir(exist_ok=True)
    os.makedirs("output", exist_ok=True)

    module = load_script("demos.py")
    for func_name, func in getmembers(module, callable):
        if not hasattr(func, "iam_a_ngl_scene_func"):
            continue

        filename = _OUTDIR / f"{func_name}.mp4"
        if filename.exists():
            print(f"{filename.name}: already present, skipping")
            continue

        cfg = SceneCfg(samples=8)
        data = func(cfg)

        export = export_workers(data, filename.as_posix(), resolution="1080p", profile_id="mp4_h264_420")
        for progress in export:
            sys.stdout.write(f"\r{filename.name}: {progress:.1f}%")
            sys.stdout.flush()
        sys.stdout.write("\n")


if __name__ == "__main__":
    _main()
