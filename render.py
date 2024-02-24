import os
import subprocess
import sys
from inspect import getmembers
from pathlib import Path

import pynopegl as ngl
from pynopegl_utils.misc import SceneCfg, get_viewport
from pynopegl_utils.module import load_script

_RES = 1080
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
        scene = data.scene

        ar = scene.aspect_ratio
        height = _RES
        width = int(height * ar[0] / ar[1])

        # make sure it's a multiple of 2 for the h264 codec
        width &= ~1

        fps = scene.framerate
        duration = scene.duration

        fd_r, fd_w = os.pipe()
        cmd = [
            # fmt: off
            "ffmpeg", "-r", "%d/%d" % fps,
            "-v", "warning",
            "-nostats", "-nostdin",
            "-f", "rawvideo",
            "-video_size", "%dx%d" % (width, height),
            "-pixel_format", "rgba",
            "-i", "pipe:%d" % fd_r,
            "-pix_fmt", "yuv420p",  # for compatibility
            "-y", filename,
            # fmt: on
        ]

        reader = subprocess.Popen(cmd, pass_fds=(fd_r,))
        os.close(fd_r)

        capture_buffer = bytearray(width * height * 4)

        ctx = ngl.Context()
        ctx.configure(
            ngl.Config(
                platform=ngl.Platform.AUTO,
                backend=ngl.Backend.AUTO,
                offscreen=True,
                width=width,
                height=height,
                viewport=get_viewport(width, height, scene.aspect_ratio),
                samples=cfg.samples,
                capture_buffer=capture_buffer,
            )
        )
        ctx.set_scene(scene)

        # Draw every frame
        nb_frame = int(duration * fps[0] / fps[1])
        for i in range(nb_frame):
            time = i * fps[1] / float(fps[0])
            ctx.draw(time)
            os.write(fd_w, capture_buffer)
            progress = i / (nb_frame - 1) * 100
            sys.stdout.write(f"\r{filename.name}: {progress:.1f}%")
            sys.stdout.flush()
        sys.stdout.write("\n")

        os.close(fd_w)
        reader.wait()


if __name__ == "__main__":
    _main()
