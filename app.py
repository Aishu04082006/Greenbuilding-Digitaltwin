"""Camera-driven live 3D Green Building Digital Twin."""
import tempfile
import time
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from ai import recommend
from brain import Controller, Predictor, device_kw, send_command
from room3d import build_live_fig
from twin import GRID_KG_PER_KWH, TARIFF_INR, Room

st.set_page_config(page_title="Green Building Digital Twin", page_icon="🌿", layout="wide")
st.title("🌿 Green Building Digital Twin - Live")
st.caption("Camera sees → twin predicts → twin switches devices off when nobody is there")

# name, capacity, lights_kw, ac_kw, equipment_kw
SPECS = [("Classroom 101", 40, 0.4, 1.5, 0.3), ("Lab 201", 30, 0.5, 2.0, 1.2),
         ("Seminar Hall", 80, 0.8, 3.5, 0.8), ("Library", 60, 0.6, 2.5, 0.4)]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📷 Cameras")
    cam_backend = st.selectbox("Webcam driver (try the other one if no picture)", ["Default", "DirectShow"])
    cam_first = st.number_input("Webcam index (try 1 if 0 fails)", 0, 5, 0)
    src, vids, urls = {}, {}, {}
    for i, s in enumerate(SPECS):
        src[i] = st.selectbox(s[0], ["No camera", "Webcam", "Video file", "Phone / IP camera"], index=1 if i == 0 else 0, key=f"src{i}")
        if src[i] == "Video file":
            vids[i] = st.file_uploader(f"{s[0]} video", type=["mp4", "avi", "mov"], key=f"vid{i}")
        elif src[i] == "Phone / IP camera":
            urls[i] = st.text_input(f"{s[0]} stream URL", key=f"url{i}", placeholder="http://192.168.x.x:8080/video")
    if sum(v == "Webcam" for v in src.values()) > 1:
        st.warning("A laptop has only ONE webcam. Use it for one room; give the other rooms a Video file or a Phone / IP camera.")
    st.header("🤖 Control & prediction")
    mode = "auto" if st.radio("Mode", ["Auto-control", "Monitor only"]) == "Auto-control" else "monitor"
    grace = st.slider("Switch OFF after room is empty for (s)", 3, 60, 8)
    auto_on = st.checkbox("Switch ON when someone enters", True)
    horizon_s = st.slider("Predict ahead (s)", 5, 120, 20)
    speed = st.slider("Time acceleration (x)", 1, 300, 60)
    duration = st.slider("Run time (s)", 30, 1800, 600)
    st.header("🔎 Detection")
    light_thr = st.slider("Lights-on brightness threshold", 40, 200, 90)
    fan_mode = st.radio("Fan status source", ["Assume ON", "Camera (blade motion)", "Assume OFF"])
    ac_mode = st.radio("AC status source", ["Assume ON", "Camera (indicator light)", "Assume OFF"])
    fan_x = st.slider("Fan area - X (%)", 0, 100, (30, 70))
    fan_y = st.slider("Fan area - Y (%)", 0, 100, (0, 30))
    ac_x = st.slider("AC area - X (%)", 0, 100, (70, 100))
    ac_y = st.slider("AC area - Y (%)", 0, 100, (0, 25))
    relay_url = st.text_input("Relay / ESP32 URL (optional)", "")
    start = st.button("▶ Start live twin", type="primary")


# ---------------- Helpers ----------------
cam_ids = {}


def open_webcam(idx):
    backend = cv2.CAP_DSHOW if cam_backend == "DirectShow" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(idx), backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):          # warm-up: first frames are often empty
            cap.read()
    return cap


def open_caps():
    caps, cam_idx = {}, int(cam_first)
    for i in range(len(SPECS)):
        cap = None
        if src[i] == "Webcam":
            cap = open_webcam(cam_idx)
            cam_ids[i] = cam_idx
            cam_idx += 1
        elif src[i] == "Phone / IP camera" and urls.get(i):
            cap = cv2.VideoCapture(urls[i])
        elif src[i] == "Video file" and vids.get(i) is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vids[i].read())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
        if cap is not None:
            if cap.isOpened():
                caps[i] = cap
            else:
                st.warning(f"Could not open the source for {SPECS[i][0]}.")
    return caps


def next_frame(cap, skip=1):
    frame = None
    for _ in range(skip):
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if not ok:
                return None
        frame = f
    return frame


def show_kpis(ph, stats):
    saved = max(0.0, stats["base"] - stats["ctrl"])
    with ph.container():
        c = st.columns(5)
        c[0].metric("Without control", f"{stats['base']:.2f} kWh")
        c[1].metric("With auto-control", f"{stats['ctrl']:.2f} kWh")
        c[2].metric("Energy saved", f"{saved:.2f} kWh")
        c[3].metric("CO₂ avoided", f"{saved * GRID_KG_PER_KWH:.2f} kg")
        c[4].metric("Money saved", f"₹{saved * TARIFF_INR:.0f}")


def chips(s):
    ac = {"on": "ON", "eco": "ECO", "off": "OFF"}[s["ac"]]
    return (f"**{s['name']}** · 👥 {s['people']} · 💡 {'ON' if s['lights'] else 'OFF'} · "
            f"🌀 {'ON' if s['fan'] else 'OFF'} · ❄️ {ac}  \n"
            f"🔮 next {s['horizon']}s: **{s['p_occ'] * 100:.0f}%** occupied (~{s['p_people']}) · "
            f"empty for {s['empty_for']:.0f}s")


# ---------------- Layout ----------------
ph_kpi = st.empty()
left, right = st.columns([1.6, 1])
ph_3d = left.empty()
ph_feed = {i: right.empty() for i in range(len(SPECS))}
ph_info = right.empty()
ph_log = right.empty()
ph_chart = st.empty()

# ---------------- Live run ----------------
if start:
    caps = open_caps()
    if not caps:
        st.error("Choose at least one camera (Webcam or an uploaded video) in the sidebar.")
        st.stop()
    for i in caps:
        ph_feed[i].info("Starting camera and loading the detection model - this can take a few seconds...")
    from vision import analyze          # lazy import (needs ultralytics)
    fan_roi = (fan_x[0] / 100, fan_y[0] / 100, fan_x[1] / 100, fan_y[1] / 100)
    ac_roi = (ac_x[0] / 100, ac_y[0] / 100, ac_x[1] / 100, ac_y[1] / 100)
    rooms = [Room(*s) for s in SPECS]
    preds, ctrls = [Predictor() for _ in SPECS], [Controller() for _ in SPECS]
    stats, log, series = {"base": 0.0, "ctrl": 0.0}, [], []
    t_start = last = time.time()
    avg_dt, n, states = 0.5, 0, []
    lost, last_state = {}, {}

    try:
        while time.time() - t_start < duration:
            now = time.time()
            dt = min(now - last, 2.0)
            last = now
            avg_dt = 0.9 * avg_dt + 0.1 * max(dt, 0.05)
            sim_min = max(dt * speed / 60, 1e-4)
            h_ticks = max(1, round(horizon_s / max(avg_dt, 0.2)))
            states, tick_base, tick_ctrl = [], 0.0, 0.0

            for i, r in enumerate(rooms):
                if i not in caps:
                    states.append(dict(name=r.name, cam=False))
                    continue
                frame = next_frame(caps[i], 1 if src[i] in ("Webcam", "Phone / IP camera") else 6)
                if frame is None:
                    lost[i] = lost.get(i, 0) + 1
                    ph_feed[i].warning(f"{r.name}: camera frame lost - reconnecting...")
                    if src[i] == "Webcam" and lost[i] % 5 == 0:
                        caps[i].release()
                        caps[i] = open_webcam(cam_ids[i])
                elif src[i] == "Phone / IP camera" and lost[i] % 5 == 0:
                    caps[i].release()
                    caps[i] = cv2.VideoCapture(urls[i])
                    states.append(last_state.get(i, dict(name=r.name, cam=False)))
                    time.sleep(0.2)
                    continue
                lost[i] = 0
                res = analyze(frame, light_thr, fan_roi, ac_roi, key=r.name)
                det = dict(people=res["people"], lights=res["lights"],
                           fan=res["fan"] if fan_mode.startswith("Camera") else fan_mode == "Assume ON",
                           ac=res["ac"] if ac_mode.startswith("Camera") else ac_mode == "Assume ON")

                preds[i].update(det["people"])
                p_occ, p_people = preds[i].forecast(h_ticks)
                cmd, changes, msg, empty_for = ctrls[i].step(now, det, p_occ, mode, auto_on, grace)
                if msg:
                    log.insert(0, f"`{datetime.now():%H:%M:%S}` **{r.name}**: {msg}")
                    for dev, val in changes:
                        send_command(relay_url, r.name, dev, val)

                occ = det["people"] > 0
                base_kw = device_kw(r, det["lights"] or occ, det["fan"] or occ,
                                    "on" if (det["ac"] or occ) else "off")
                ctrl_kw = device_kw(r, cmd["lights"], cmd["fan"], cmd["ac"])
                stats["base"] += base_kw * sim_min / 60
                stats["ctrl"] += ctrl_kw * sim_min / 60
                tick_base += base_kw
                tick_ctrl += ctrl_kw
                r.update(det["people"], det["lights"], det["ac"], sim_min, det["fan"])

                wasting = det["people"] == 0 and (cmd["lights"] or cmd["fan"] or cmd["ac"] == "on")
                states.append(dict(name=r.name, cam=True, centers=res["centers"], people=det["people"],
                                   lights=cmd["lights"], fan=cmd["fan"], ac=cmd["ac"], det=det, p_occ=p_occ,
                                   p_people=p_people, empty_for=empty_for, waste=wasting, horizon=horizon_s))
                last_state[i] = states[-1]
                ph_feed[i].image(res["annotated"], channels="BGR", caption=f"{r.name} - camera")

            n += 1
            ph_3d.plotly_chart(build_live_fig(states, n), use_container_width=True)
            show_kpis(ph_kpi, stats)
            with ph_info.container():
                for s in states:
                    if s.get("cam"):
                        st.markdown(chips(s))
            with ph_log.container():
                st.markdown("**📝 Control log**")
                for line in log[:8] or ["No actions yet"]:
                    st.markdown(line)
            series.append({"Without control (kW)": tick_base, "With auto-control (kW)": tick_ctrl})
            ph_chart.line_chart(pd.DataFrame(series))

    finally:
        for c in caps.values():
            c.release()
        st.session_state.update(live_rooms=rooms, live_stats=stats, live_fig=build_live_fig(states, n),
                                live_log=log, live_series=series)
    st.success("Run finished. Press ▶ Start live twin to run again.")
elif "live_rooms" in st.session_state:
    show_kpis(ph_kpi, st.session_state.live_stats)
    ph_3d.plotly_chart(st.session_state.live_fig, use_container_width=True)
    with ph_log.container():
        st.markdown("**📝 Control log**")
        for line in st.session_state.live_log[:8] or ["No actions yet"]:
            st.markdown(line)
    ph_chart.line_chart(pd.DataFrame(st.session_state.live_series))
else:
    st.info("Pick a camera in the sidebar and press **▶ Start live twin**. Walk out of the frame and watch the "
            "lights, fan and AC switch off in the 3D building.")

# ---------------- After the run ----------------
if "live_rooms" in st.session_state:
    rooms = st.session_state.live_rooms
    st.subheader("🤖 GenAI recommendations")
    if st.button("Generate recommendations"):
        with st.spinner("Analysing..."):
            st.session_state.reco = recommend([r for r in rooms if r.history])
    if "reco" in st.session_state:
        st.markdown(st.session_state.reco)
    rows = [x for r in rooms for x in r.history]
    if rows:
        st.download_button("⬇ Download data (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           "twin_data.csv", "text/csv")
