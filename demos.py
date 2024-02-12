import array
import platform
from pathlib import Path
from textwrap import dedent

import pynopegl as ngl
from pynopegl_utils.misc import load_media

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_IMG_CITY = (_ASSETS_DIR / "city.jpg").as_posix()
_IMG_TORII = (_ASSETS_DIR / "torii-gate.jpg").as_posix()

_VID_BBB = (_ASSETS_DIR / "bbb.mp4").as_posix()
_VID_PIPER = (_ASSETS_DIR / "piper.mp4").as_posix()
_VID_ROYAUME = (_ASSETS_DIR / "le-royaume.mp4").as_posix()

_FONT_SHIPPORI = ngl.FontFace((_ASSETS_DIR / "ShipporiMincho-Regular.ttf").as_posix())
_FONT_UBUNTU = ngl.FontFace((_ASSETS_DIR / "Ubuntu-Light.ttf").as_posix())


@ngl.scene(compat_specs="~=0.11")
def audiotex(cfg: ngl.SceneCfg):
    media = load_media(_VID_ROYAUME)
    cfg.duration = media.duration
    cfg.aspect_ratio = (media.width, media.height)

    q = ngl.Quad((-1, -1, 0), (2, 0, 0), (0, 2, 0))

    audio_m = ngl.Media(media.filename, audio_tex=True)
    audio_tex = ngl.Texture2D(data_src=audio_m, mag_filter="nearest", min_filter="nearest")

    video_m = ngl.Media(media.filename)
    video_tex = ngl.Texture2D(data_src=video_m)

    vertex = dedent(
        """\
        void main()
        {
            ngl_out_pos = ngl_projection_matrix * ngl_modelview_matrix * vec4(ngl_position, 1.0);
            var_tex0_coord = (tex0_coord_matrix * vec4(ngl_uvcoord, 0.0, 1.0)).xy;
            var_tex1_coord = (tex1_coord_matrix * vec4(ngl_uvcoord, 0.0, 1.0)).xy;
        }
        """
    )

    fragment = dedent(
        """\
        float wave(float amp, float y, float yoff)
        {
            float s = (amp + 1.0) / 2.0; // [-1;1] -> [0;1]
            float v = yoff + s/4.0;         // [0;1] -> [off;off+0.25]
            return smoothstep(v-0.005, v, y)
                 - smoothstep(v, v+0.005, y);
        }

        float freq(float power, float y, float yoff)
        {
            float p = sqrt(power);
            float v = clamp(p, 0.0, 1.0) / 4.0; // [0;+oo] -> [0;0.25]
            float a = yoff + 0.25;
            float b = a - v;
            return step(y, a) * (1.0 - step(y, b)); // y <= a && y > b
        }

        void main()
        {
            int freq_line = 2                          // skip the 2 audio channels
                          + (10 - freq_precision) * 2; // 2x10 lines of FFT
            float fft1 = float(freq_line) + 0.5;
            float fft2 = float(freq_line) + 1.5;
            float x = var_tex0_coord.x;
            float y = var_tex0_coord.y;
            vec4 video_pix = ngl_texvideo(tex1, var_tex1_coord);
            vec2 sample_id_ch_1 = vec2(x,  0.5 / 22.);
            vec2 sample_id_ch_2 = vec2(x,  1.5 / 22.);
            vec2  power_id_ch_1 = vec2(x, fft1 / 22.);
            vec2  power_id_ch_2 = vec2(x, fft2 / 22.);
            float sample_ch_1 = texture(tex0, sample_id_ch_1).x;
            float sample_ch_2 = texture(tex0, sample_id_ch_2).x;
            float  power_ch_1 = texture(tex0,  power_id_ch_1).x;
            float  power_ch_2 = texture(tex0,  power_id_ch_2).x;
            float wave1 = wave(sample_ch_1, y, 0.0);
            float wave2 = wave(sample_ch_2, y, 0.25);
            float freq1 = freq(power_ch_1, y, 0.5);
            float freq2 = freq(power_ch_2, y, 0.75);
            vec3 audio_pix = vec3(0.0, 1.0, 0.5) * wave2
                           + vec3(0.5, 1.0, 0.0) * wave1
                           + vec3(1.0, 0.5, 0.0) * freq1
                           + vec3(1.0, 0.0, 0.5) * freq2;
            ngl_out_color = mix(video_pix, vec4(audio_pix, 1.0), overlay);
        }
        """
    )

    p = ngl.Program(vertex=vertex, fragment=fragment)
    p.update_vert_out_vars(var_tex0_coord=ngl.IOVec2(), var_tex1_coord=ngl.IOVec2())
    render = ngl.Render(q, p)
    render.update_frag_resources(tex0=audio_tex, tex1=video_tex)
    render.update_frag_resources(overlay=ngl.UniformFloat(0.6, live_id="overlay", live_min=0, live_max=1))
    render.update_frag_resources(freq_precision=ngl.UniformInt(7, live_id="freq_precision", live_min=1, live_max=10))
    return render


@ngl.scene(compat_specs="~=0.11")
def compositing(cfg: ngl.SceneCfg):
    cfg.aspect_ratio = (1, 1)
    cfg.duration = 6

    vertex = dedent(
        """\
        void main()
        {
            ngl_out_pos = ngl_projection_matrix * ngl_modelview_matrix * vec4(ngl_position, 1.0);
            uv = (ngl_uvcoord - .5) * 2.;
        }
        """
    )

    fragment = dedent(
        """\
        void main() {
            float sd = length(uv + off) - 0.5; // signed distance to a circle of radius 0.5
            float alpha = clamp(0.5 - sd / fwidth(sd), 0.0, 1.0); // anti-aliasing
            ngl_out_color = vec4(color, 1.0) * alpha;
        }
        """
    )

    # We can not use a circle geometry because the whole areas must be
    # rasterized for the compositing to work, so instead we build 2 overlapping
    # quad into which we draw colored circles, offsetted with an animation.
    # Alternatively, we could use a RTT.
    quad = ngl.Quad(corner=(-1, -1, 0), width=(2, 0, 0), height=(0, 2, 0))
    prog = ngl.Program(vertex=vertex, fragment=fragment)
    prog.update_vert_out_vars(uv=ngl.IOVec2())

    A_off_kf = (
        ngl.AnimKeyFrameVec2(0, (-1 / 3, 0)),
        ngl.AnimKeyFrameVec2(cfg.duration / 2, (1 / 3, 0)),
        ngl.AnimKeyFrameVec2(cfg.duration, (-1 / 3, 0)),
    )
    B_off_kf = (
        ngl.AnimKeyFrameVec2(0, (1 / 3, 0)),
        ngl.AnimKeyFrameVec2(cfg.duration / 2, (-1 / 3, 0)),
        ngl.AnimKeyFrameVec2(cfg.duration, (1 / 3, 0)),
    )
    A_off = ngl.AnimatedVec2(A_off_kf)
    B_off = ngl.AnimatedVec2(B_off_kf)

    scenes = []
    operators = ["src_over", "dst_over", "src_out", "dst_out", "src_in", "dst_in", "src_atop", "dst_atop", "xor"]

    for op in operators:
        A = ngl.Render(quad, prog, label="A")
        A.update_frag_resources(color=ngl.UniformVec3(value=(0.0, 0.5, 1.0)), off=A_off)

        B = ngl.Render(quad, prog, label="B", blending=op)
        B.update_frag_resources(color=ngl.UniformVec3(value=(1.0, 0.5, 0.0)), off=B_off)

        bg = ngl.RenderColor(blending="dst_over")

        # draw A in current FBO, then draw B with the current operator, and
        # then result goes over the white background
        ret = ngl.Group(children=(A, B, bg))

        label_h = 1 / 4
        label_pad = 0.1
        label = ngl.Text(
            op,
            fg_color=(0, 0, 0),
            bg_color=(0.8, 0.8, 0.8),
            bg_opacity=1,
            box_corner=(label_pad / 2 - 1, 1 - label_h - label_pad / 2, 0),
            box_width=(2 - label_pad, 0, 0),
            box_height=(0, label_h, 0),
        )
        ret.add_children(label)

        scenes.append(ret)

    return ngl.GridLayout(scenes, size=(3, 3))


@ngl.scene(compat_specs="~=0.11", controls=dict(dim=ngl.scene.Range(range=[1, 50])))
def cropboard(cfg: ngl.SceneCfg, dim=32):
    m0 = load_media(_VID_BBB)
    cfg.duration = 10
    cfg.aspect_ratio = (m0.width, m0.height)

    kw = kh = 1.0 / dim
    qw = qh = 2.0 / dim

    vertex = dedent(
        """\
        void main()
        {
            vec4 position = vec4(ngl_position, 1.0) + vec4(mix(translate_a, translate_b, time), 0.0, 0.0);
            ngl_out_pos = ngl_projection_matrix * ngl_modelview_matrix * position;
            var_tex0_coord = (tex0_coord_matrix * vec4(ngl_uvcoord + uv_offset, 0.0, 1.0)).xy;
        }
        """
    )

    fragment = dedent(
        """\
        void main()
        {
            ngl_out_color = ngl_texvideo(tex0, var_tex0_coord);
        }
        """
    )
    p = ngl.Program(vertex=vertex, fragment=fragment)
    p.update_vert_out_vars(var_tex0_coord=ngl.IOVec2(), var_uvcoord=ngl.IOVec2())
    m = ngl.Media(m0.filename)
    t = ngl.Texture2D(data_src=m)

    uv_offset_buffer = array.array("f")
    translate_a_buffer = array.array("f")
    translate_b_buffer = array.array("f")

    q = ngl.Quad(
        corner=(0, 0, 0),
        width=(qw, 0, 0),
        height=(0, qh, 0),
        uv_corner=(0, 0),
        uv_width=(kw, 0),
        uv_height=(0, kh),
    )

    for y in range(dim):
        for x in range(dim):
            uv_offset = [x * kw, (y + 1.0) * kh - 1.0]
            src = [cfg.rng.uniform(-2, 2), cfg.rng.uniform(-2, 2)]
            dst = [x * qw - 1.0, 1.0 - (y + 1.0) * qh]

            uv_offset_buffer.extend(uv_offset)
            translate_a_buffer.extend(src)
            translate_b_buffer.extend(dst)

    utime_animkf = [
        ngl.AnimKeyFrameFloat(0, 0),
        ngl.AnimKeyFrameFloat(cfg.duration - 1, 1, "exp_out"),
    ]
    utime = ngl.AnimatedFloat(utime_animkf)

    render = ngl.Render(q, p, nb_instances=dim**2)
    render.update_frag_resources(tex0=t)
    render.update_vert_resources(time=utime)
    render.update_instance_attributes(
        uv_offset=ngl.BufferVec2(data=uv_offset_buffer),
        translate_a=ngl.BufferVec2(data=translate_a_buffer),
        translate_b=ngl.BufferVec2(data=translate_b_buffer),
    )
    return render


@ngl.scene(compat_specs="~=0.11", controls=dict(n=ngl.scene.Range(range=[2, 10])))
def fibo(cfg: ngl.SceneCfg, n=8):
    cfg.duration = 5.0
    cfg.aspect_ratio = (1, 1)

    fib = [0, 1, 1]
    for i in range(2, n):
        fib.append(fib[i] + fib[i - 1])
    fib = fib[::-1]

    shift = 1 / 3.0  # XXX: what's the exact math here?
    shape_scale = 1.0 / ((2.0 - shift) * sum(fib))

    orig = (-shift, -shift, 0)
    g = None
    root = None
    for i, x in enumerate(fib[:-1]):
        w = x * shape_scale
        gray = 1.0 - i / float(n)
        color = (gray, gray, gray)
        q = ngl.Quad(orig, (w, 0, 0), (0, w, 0))
        render = ngl.RenderColor(color, geometry=q)

        new_g = ngl.Group()
        animkf = [
            ngl.AnimKeyFrameFloat(0, 90),
            ngl.AnimKeyFrameFloat(cfg.duration / 2, -90, "exp_in_out"),
            ngl.AnimKeyFrameFloat(cfg.duration, 90, "exp_in_out"),
        ]
        rot = ngl.Rotate(new_g, anchor=orig, angle=ngl.AnimatedFloat(animkf))
        if g:
            g.add_children(rot)
        else:
            root = rot
        g = new_g
        new_g.add_children(render)
        orig = (orig[0] + w, orig[1] + w, 0)

    assert root is not None
    return root


@ngl.scene(compat_specs="~=0.11")
def japanese_haiku(cfg):
    m0 = load_media(_IMG_TORII)
    cfg.duration = 9.0
    cfg.aspect_ratio = (m0.width, m0.height)

    bgalpha_animkf = [
        ngl.AnimKeyFrameFloat(0, 0.0),
        ngl.AnimKeyFrameFloat(1, 0.4),
        ngl.AnimKeyFrameFloat(cfg.duration - 1, 0.4),
        ngl.AnimKeyFrameFloat(cfg.duration, 0.0),
    ]
    bg_filter = ngl.RenderColor(color=(0, 0, 0), opacity=ngl.AnimatedFloat(bgalpha_animkf), blending="src_over")

    media = ngl.Media(m0.filename)
    tex = ngl.Texture2D(data_src=media)
    bg = ngl.RenderTexture(tex)

    text = ngl.Text(
        text="減る記憶、\nそれでも増える、\nパスワード",
        font_faces=[_FONT_SHIPPORI],
        fg_color=(1.0, 0.8, 0.6),
        bg_opacity=0.0,
        font_scale=0.6,
        box_height=(0, 1.3, 0),
        writing_mode="vertical-rl",
        valign="top",
        aspect_ratio=cfg.aspect_ratio,
        effects=[
            ngl.TextEffect(
                target="text",
                start=0.0,
                end=cfg.duration,
            ),
            ngl.TextEffect(
                start=1.0,
                end=cfg.duration - 3.0,
                target="char",
                overlap=0.7,
                opacity=ngl.AnimatedFloat(
                    [
                        ngl.AnimKeyFrameFloat(0, 0),
                        ngl.AnimKeyFrameFloat(1, 1),
                    ]
                ),
            ),
            ngl.TextEffect(
                target="text",
                start=cfg.duration - 2.0,
                end=cfg.duration - 1.0,
                opacity=ngl.AnimatedFloat(
                    [
                        ngl.AnimKeyFrameFloat(0, 1),
                        ngl.AnimKeyFrameFloat(1, 0),
                    ]
                ),
            ),
        ],
    )

    text = ngl.TimeRangeFilter(text, start=1, end=cfg.duration - 1)
    return ngl.Group(children=(bg, bg_filter, text))


@ngl.scene(compat_specs="~=0.11", controls=dict(bg_file=ngl.scene.File()))
def prototype(cfg, bg_file=_IMG_CITY):
    m0 = load_media(bg_file)
    cfg.aspect_ratio = (m0.width, m0.height)

    delay = 1.5  # delay before looping
    pause = 2.0
    text_effect_duration = 4.0

    in_start = 0
    in_end = in_start + text_effect_duration
    out_start = in_end + pause
    out_end = out_start + text_effect_duration
    cfg.duration = out_end + delay

    opacityin_animkf = [ngl.AnimKeyFrameFloat(0, 0), ngl.AnimKeyFrameFloat(1, 1)]
    opacityout_animkf = [ngl.AnimKeyFrameFloat(0, 1), ngl.AnimKeyFrameFloat(1, 0)]
    blurin_animkf = [ngl.AnimKeyFrameFloat(0, 1), ngl.AnimKeyFrameFloat(1, 0)]
    blurout_animkf = [ngl.AnimKeyFrameFloat(0, 0), ngl.AnimKeyFrameFloat(1, 1)]

    text_effect_settings = dict(
        target="char",
        random=True,
        overlap=0.7,
    )

    text = ngl.Text(
        text="Prototype",
        live_id="text",
        bg_opacity=0,
        font_faces=[_FONT_UBUNTU],
        aspect_ratio=cfg.aspect_ratio,
        effects=[
            ngl.TextEffect(
                start=in_start,
                end=in_end,
                random_seed=6,
                opacity=ngl.AnimatedFloat(opacityin_animkf),
                blur=ngl.AnimatedFloat(blurin_animkf),
                **text_effect_settings
            ),
            ngl.TextEffect(
                start=out_start,
                end=out_end,
                random_seed=2,
                opacity=ngl.AnimatedFloat(opacityout_animkf),
                blur=ngl.AnimatedFloat(blurout_animkf),
                **text_effect_settings
            ),
        ],
    )

    text_scale = [0.3, 1.0]
    text_animkf = [
        ngl.AnimKeyFrameVec3(0.0, [text_scale[0]] * 3),
        ngl.AnimKeyFrameVec3(cfg.duration, [text_scale[1]] * 3),
    ]
    text = ngl.Scale(text, factors=ngl.AnimatedVec3(text_animkf))
    text = ngl.TimeRangeFilter(text, start=0, end=out_end)

    media = ngl.Media(m0.filename)
    tex = ngl.Texture2D(data_src=media)
    bg = ngl.RenderTexture(tex)

    bg_scale = [1.0, 1.4]
    bg_animkf = [
        ngl.AnimKeyFrameVec3(0.0, [bg_scale[0]] * 3),
        ngl.AnimKeyFrameVec3(cfg.duration, [bg_scale[1]] * 3),
    ]
    bg = ngl.Scale(bg, factors=ngl.AnimatedVec3(bg_animkf))

    return ngl.Group(children=(bg, text))


@ngl.scene(compat_specs="~=0.11", controls=dict(source=ngl.scene.File()))
def scopes(cfg, source=_VID_PIPER):
    # FIXME this check is not sufficient when cross-building a scene
    if platform.system() == "Darwin" and cfg.backend == "opengl":
        cfg.aspect_ratio = (1, 1)
        return ngl.Text("macOS OpenGL\nimplementation\ndoesn't support\ncompute shaders\n:(", fg_color=(1, 0.3, 0.3))

    m = load_media(source)
    cfg.duration = m.duration
    cfg.aspect_ratio = (m.width, m.height)

    texture = ngl.Texture2D(data_src=ngl.Media(m.filename))
    stats = ngl.ColorStats(texture)
    scenes = [
        ngl.RenderTexture(texture),
        ngl.RenderWaveform(stats=stats, mode="parade"),
        ngl.RenderWaveform(stats=stats, mode="mixed"),
        ngl.RenderHistogram(stats=stats, mode="parade"),
    ]
    return ngl.GridLayout(scenes, size=(2, 2))
