"""The 'brain' of the twin: occupancy prediction + automatic device controller."""
import numpy as np

AC_FACTOR = {"on": 1.0, "eco": 0.4, "off": 0.0}


class Predictor:
    """Learns from the camera's people-count stream, live.
    - Holt smoothing (level + trend) forecasts how many people will be there.
    - A 2-state Markov chain (empty/occupied), learned from the camera, gives P(occupied in h steps)."""

    def __init__(self, alpha=0.4, beta=0.2):
        self.alpha, self.beta = alpha, beta
        self.level, self.trend, self.prev = None, 0.0, None
        self.counts = np.array([[3.0, 1.0], [1.0, 3.0]])   # prior: rooms tend to stay as they are

    def update(self, people):
        occ = int(people > 0)
        if self.prev is not None:
            self.counts[self.prev, occ] += 1
        self.prev = occ
        if self.level is None:
            self.level = float(people)
        else:
            last = self.level
            self.level = self.alpha * people + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - last) + (1 - self.beta) * self.trend

    def forecast(self, h):
        if self.prev is None:
            return 0.5, 0
        P = self.counts / self.counts.sum(axis=1, keepdims=True)
        p_occ = float(np.linalg.matrix_power(P, max(1, int(h)))[self.prev, 1])
        n = max(0.0, self.level + h * self.trend)
        return p_occ, (int(round(n)) if p_occ >= 0.5 else 0)


class Controller:
    """Decides what the twin commands each device to do, from camera + prediction."""

    def __init__(self):
        self.empty_since, self.cmd = None, None

    def step(self, now, det, p_return, mode="auto", auto_on=True, grace=8.0, precool_at=0.6):
        people = det["people"]
        if people > 0:
            self.empty_since = None
        elif self.empty_since is None:
            self.empty_since = now
        empty_for = 0.0 if people > 0 else now - self.empty_since

        base = dict(lights=bool(det["lights"]), fan=bool(det["fan"]), ac="on" if det["ac"] else "off")
        if self.cmd is None:
            self.cmd = dict(base)
        new, why = dict(self.cmd), ""
        if mode != "auto":                      # monitor only: twin mirrors what the camera sees
            new = dict(base)
        elif people > 0:
            if auto_on:
                new, why = dict(lights=True, fan=True, ac="on"), "person detected"
            else:
                new = dict(base)
        elif empty_for >= grace:
            new["lights"], new["fan"] = False, False
            if p_return >= precool_at:
                new["ac"] = "eco"
                why = f"empty {empty_for:.0f}s, return likely {p_return:.0%} -> AC kept in ECO"
            else:
                new["ac"] = "off"
                why = f"empty {empty_for:.0f}s, return unlikely ({p_return:.0%})"

        changes = [(d, new[d]) for d in ("lights", "fan", "ac") if new[d] != self.cmd[d]]
        msg = ""
        if changes:
            def lab(v):
                return {True: "ON", False: "OFF"}.get(v, str(v).upper())
            msg = ", ".join(f"{d} {lab(v)}" for d, v in changes) + (f"  ({why})" if why else "")
        self.cmd = new
        return new, changes, msg, empty_for


def device_kw(room, lights, fan, ac):
    return room.lights_kw * bool(lights) + room.fan_kw * bool(fan) + room.ac_kw * AC_FACTOR[ac]


def send_command(url, room_name, device, state):
    """Optional: tell a real relay / ESP32. Expects e.g. GET <url>/<room>/<device>/<state>."""
    if not url:
        return
    try:
        import requests
        requests.get(f"{url.rstrip('/')}/{room_name.replace(' ', '_')}/{device}/{str(state).lower()}", timeout=1)
    except Exception:
        pass
