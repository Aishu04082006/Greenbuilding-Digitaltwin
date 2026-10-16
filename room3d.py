"""Live 3D digital twin. Everything shown is driven by camera detections + the auto-controller."""
import math
import plotly.graph_objects as go

W, D, H, GAP, FG = 2.4, 2.0, 1.2, 0.3, 0.3


def geom(i):
    floor, col = divmod(i, 2)
    x0 = col * (W + GAP)
    z0 = floor * (H + FG)
    return x0, x0 + W, 0.0, D, z0, z0 + H


def _box(x0, x1, y0, y1, z0, z1, color, opacity, hover=None):
    X = [x0, x1, x1, x0, x0, x1, x1, x0]
    Y = [y0, y0, y1, y1, y0, y0, y1, y1]
    Z = [z0, z0, z0, z0, z1, z1, z1, z1]
    I = [0, 0, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3]
    J = [1, 2, 5, 6, 1, 5, 3, 7, 2, 6, 0, 4]
    K = [2, 3, 6, 7, 5, 4, 7, 6, 6, 5, 4, 7]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=color, opacity=opacity, flatshading=True,
                     hoverinfo="text" if hover else "skip", hovertext=hover, showlegend=False)


def _edges(x0, x1, y0, y1, z0, z1, color, width=4):
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)]
    xs, ys, zs = [], [], []
    for a, b in zip(pts, pts[1:]):
        xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    for (x, y) in ((x1, y0), (x1, y1), (x0, y1)):
        xs += [x, x, None]; ys += [y, y, None]; zs += [z0, z1, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", showlegend=False)


def _floor(x0, x1, y0, y1, z, color, opacity=0.9):
    return go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[z] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=opacity, hoverinfo="skip", showlegend=False)


def _cone(cx, cy, zt, zb, r, n=14):
    xs = [cx] + [cx + r * math.cos(2 * math.pi * k / n) for k in range(n)]
    ys = [cy] + [cy + r * math.sin(2 * math.pi * k / n) for k in range(n)]
    zs = [zt] + [zb] * n
    return go.Mesh3d(x=xs, y=ys, z=zs, i=[0] * n, j=[1 + k for k in range(n)],
                     k=[1 + (k + 1) % n for k in range(n)], color="#fff176", opacity=0.16,
                     hoverinfo="skip", showlegend=False)


def _text(x, y, z, text, color="#111", size=11):
    return go.Scatter3d(x=[x], y=[y], z=[z], mode="text", text=[text], textfont=dict(size=size, color=color),
                        hoverinfo="skip", showlegend=False)


def build_live_fig(states, tick=0):
    phase, ang = (tick * 0.17) % 1.0, tick * 0.9
    data = []
    for i, s in enumerate(states):
        x0, x1, y0, y1, z0, z1 = geom(i)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        if not s.get("cam"):                                   # room without a camera: ghost
            data += [_box(x0, x1, y0, y1, z0, z1, "#90a4ae", 0.04), _edges(x0, x1, y0, y1, z0, z1, "#b0bec5", 2),
                     _text(cx, cy, z0 + 0.4, f"{s['name']} (no camera)", "#78909c")]
            continue

        on_l, on_f, ac = s["lights"], s["fan"], s["ac"]
        shell = "#c62828" if s["waste"] else "#2e7d32"
        det = s["det"]
        hover = (f"<b>{s['name']}</b><br>People (camera): {s['people']}"
                 f"<br>Commanded: lights {'ON' if on_l else 'OFF'}, fan {'ON' if on_f else 'OFF'}, AC {ac.upper()}"
                 f"<br>Camera sees: lights {'ON' if det['lights'] else 'off'}, fan {'ON' if det['fan'] else 'off'}, "
                 f"AC {'ON' if det['ac'] else 'off'}<br>Next {s['horizon']}s: {s['p_occ'] * 100:.0f}% occupied "
                 f"(~{s['p_people']} people)<br>Empty for {s['empty_for']:.0f}s")
        data += [_box(x0, x1, y0, y1, z0, z1, shell, 0.07, hover), _edges(x0, x1, y0, y1, z0, z1, shell),
                 _floor(x0, x1, y0, y1, z0, "#fff59d" if on_l else "#263238")]

        # lights: lamps + glowing cones
        lamps = [(x0 + W * 0.3, cy), (x0 + W * 0.7, cy)]
        if on_l:
            data += [_cone(lx, ly, z1 - 0.04, z0 + 0.02, 0.65) for lx, ly in lamps]
        data.append(go.Scatter3d(x=[l[0] for l in lamps], y=[l[1] for l in lamps], z=[z1 - 0.03] * 2,
                                 mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=7, symbol="square", color="#ffd600" if on_l else "#455a64",
                                             line=dict(width=1, color="black"))))
        # fan: spinning blades when ON
        hz = z1 - 0.08
        a0 = ang if on_f else 0.4
        xs, ys, zs = [], [], []
        for k in range(4):
            a = a0 + k * math.pi / 2
            xs += [cx, cx + 0.55 * math.cos(a), None]; ys += [cy, cy + 0.55 * math.sin(a), None]; zs += [hz, hz, None]
        data.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=8, color="#00e5ff" if on_f else "#78909c")))
        # AC: wall unit + airflow
        acx, acy, acz = cx, y1 - 0.12, z1 - 0.3
        data.append(_box(acx - 0.45, acx + 0.45, acy - 0.09, acy + 0.09, acz - 0.11, acz + 0.11, "#eceff1", 0.95))
        led = {"on": "#2979ff", "eco": "#ffab00", "off": "#616161"}[ac]
        data.append(go.Scatter3d(x=[acx + 0.35], y=[acy - 0.1], z=[acz], mode="markers", hoverinfo="skip",
                                 showlegend=False, marker=dict(size=6, color=led)))
        if ac != "off":
            px, py, pz = [], [], []
            for sx in (-1, 0, 1):
                for k in range(5):
                    t = ((k / 5) + phase) % 1
                    px.append(acx + sx * 0.4 * t); py.append(acy - 0.12 - t * 1.15); pz.append(acz - 0.05 - t * 0.5)
            data.append(go.Scatter3d(x=px, y=py, z=pz, mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=4, color="rgba(41,121,255,0.55)" if ac == "on"
                                                 else "rgba(255,171,0,0.6)")))
        # people: placed where the camera sees them
        bx, by, bz, hx, hy, hzs = [], [], [], [], [], []
        for (u, v) in s["centers"][:12]:
            u, v = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
            px_, py_ = x0 + 0.2 + u * (W - 0.4), y0 + 0.2 + (1 - v) * (D - 0.4)
            bx += [px_, px_, None]; by += [py_, py_, None]; bz += [z0 + 0.02, z0 + 0.5, None]
            hx.append(px_); hy.append(py_); hzs.append(z0 + 0.6)
        if hx:
            data.append(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(width=9, color="#1565c0")))
            data.append(go.Scatter3d(x=hx, y=hy, z=hzs, mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=8, color="#ffcc80", line=dict(width=1, color="#1565c0"))))
        status = f"{s['people']} people | next {s['horizon']}s: {s['p_occ'] * 100:.0f}% occupied"
        data += [_text(cx, cy, z1 + 0.2, status, "#b71c1c" if s["waste"] else "#1b5e20"),
                 _text(cx, y0 - 0.25, z0, s["name"], "#111", 12)]

    fig = go.Figure(data=data)
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=10, b=30), showlegend=False, uirevision="live",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data", camera=dict(eye=dict(x=1.6, y=-2.3, z=1.2))),
        annotations=[dict(text="Yellow cone = lights ON | cyan blades = fan ON | blue/amber air = AC on/eco | "
                               "blue figures = people seen by camera | red frame = energy being wasted",
                          x=0.5, y=0, xref="paper", yref="paper", showarrow=False, font=dict(size=11))])
    return fig
