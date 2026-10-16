"""GenAI Building Copilot: answers from live twin data AND issues device commands.
Commands are understood by a rule parser (works offline, always reliable); free-form questions go to the
LLM when ANTHROPIC_API_KEY is set, otherwise to data-driven rule answers."""
import json
import os
import re

from twin import GRID_KG_PER_KWH, TARIFF_INR

_ACT = r"\b(switch|turn|put|set|shut|kill|dim|enable|disable|resume|restore|brighten)\b"


def _rooms_in(t, rooms):
    if re.search(r"\b(all|every|everything|building|whole)\b", t):
        return list(range(len(rooms)))
    return [i for i, r in enumerate(rooms)
            if any(re.search(rf"\b{re.escape(w)}\b", t) for w in r.name.lower().split() if len(w) >= 3 or w.isdigit())]


def _devices_in(t):
    out = []
    if re.search(r"\b(lights?|lamps?|bulbs?|lighting)\b", t):
        out.append("lights")
    if re.search(r"\bfans?\b", t):
        out.append("fan")
    if re.search(r"\b(a/?c|aircon|air[- ]?conditioner(s)?|cooling)\b", t):
        out.append("ac")
    if not out and re.search(r"\b(everything|all devices)\b", t):
        out = ["lights", "fan", "ac"]
    return out


def _seconds(t, default=300):
    m = re.search(r"for\s+(\d+)\s*(sec|second|min|minute|hour|hr)", t)
    if not m:
        return default
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)[0]]


def _mins(sec):
    return f"{sec // 60} min" if sec >= 60 else f"{sec} s"


def rule_commands(text, rooms):
    """-> list of command dicts, 'NEED_ROOM', or None if the text is not a command."""
    t = text.lower()
    if not re.search(_ACT, t):
        return None
    idx, secs = _rooms_in(t, rooms), _seconds(t)
    if re.search(r"\b(resume|restore|back)\b", t) and re.search(r"\bauto", t):
        return [dict(room=i, device="auto", state=None, seconds=0) for i in (idx or range(len(rooms)))]
    devs = _devices_in(t)
    if not devs:
        return None
    if not idx:
        return "NEED_ROOM"
    pct = re.search(r"(\d{1,3})\s*%", t) or re.search(r"\bdim\b.*?\b(\d{1,3})\b", t)
    cmds = []
    for i in idx:
        for d in devs:
            if d == "lights" and (re.search(r"\bdim\b", t) or pct):
                p = int(pct.group(1)) if pct else 50
                cmds.append(dict(room=i, device="dim", state=max(0, min(100, p)) / 100, seconds=secs))
                continue
            if d == "ac" and re.search(r"\beco\b", t):
                st = "eco"
            elif re.search(r"\b(off|disable|shut|kill)\b", t):
                st = "off"
            elif re.search(r"\b(on|enable)\b", t):
                st = "on"
            else:
                continue
            cmds.append(dict(room=i, device=d, state=st, seconds=secs))
    return cmds or None


def describe(c, rooms):
    name = rooms[c["room"]].name
    if c["device"] == "auto":
        return f"✅ {name}: control handed back to **auto**."
    if c["device"] == "dim":
        return f"✅ {name}: lights set to **{c['state'] * 100:.0f}%** for {_mins(c['seconds'])} (manual override)."
    label = {"lights": "lights", "fan": "fan", "ac": "AC"}[c["device"]]
    return (f"✅ {name}: {label} switched **{str(c['state']).upper()}** for {_mins(c['seconds'])} "
            f"(manual override - say *resume auto* to release it).")


# ---------------- analytics from live data ----------------
def waste_breakdown(r):
    h = r.history
    if not h:
        return None
    emp = [x for x in h if x["people"] == 0]
    l = sum(r.lights_kw * x["hours"] for x in emp if x["lights_on"])
    f = sum(r.fan_kw * x["hours"] for x in emp if x["fan_on"])
    a = sum(r.ac_kw * x["hours"] for x in emp if x["ac_on"])
    return dict(total_h=sum(x["hours"] for x in h), empty_h=sum(x["hours"] for x in emp),
                lights=l, fan=f, ac=a, waste=l + f + a, peak=max(x["people"] for x in h))


def explain_room(i, ctx):
    r = ctx["rooms"][i]
    w = waste_breakdown(r)
    if not w:
        return f"**{r.name}** has no camera data in this run, so I can't judge its energy use."
    pr = ctx.get("per_room", {}).get(i, {})
    saved = max(0.0, pr.get("base", 0) - pr.get("ctrl", 0))
    if w["waste"] < 0.005:
        return (f"**{r.name}** was efficient: devices were not left running while it was empty "
                f"(peak occupancy {w['peak']}). Auto-control saved {saved:.2f} kWh.")
    top = max(("lights", "fan", "AC"), key=lambda k: w[k.lower()])
    pct = 100 * w["empty_h"] / max(w["total_h"], 1e-9)
    return (f"**{r.name}** was empty for {w['empty_h'] * 60:.0f} of {w['total_h'] * 60:.0f} simulated minutes "
            f"({pct:.0f}%), but the camera saw devices still running: lights {w['lights']:.2f} kWh, "
            f"fan {w['fan']:.2f} kWh, AC {w['ac']:.2f} kWh. That is **{w['waste']:.2f} kWh** wasted "
            f"(₹{w['waste'] * TARIFF_INR:.0f}, {w['waste'] * GRID_KG_PER_KWH:.2f} kg CO₂). Biggest cause: **{top}**. "
            f"Peak occupancy was {w['peak']}. Auto-control saved {saved:.2f} kWh of it.")


def rank_rooms(ctx):
    rows = [(r.name, waste_breakdown(r)) for r in ctx["rooms"]]
    rows = sorted([(n, w["waste"]) for n, w in rows if w], key=lambda x: -x[1])
    if not rows:
        return "No camera data yet."
    return "Waste ranking (devices running in an empty room):\n" + "\n".join(f"{k + 1}. {n}: {w:.2f} kWh" for k, (n, w) in enumerate(rows))


def status_text(ctx, only=None):
    out = []
    for i, s in enumerate(ctx["states"]):
        if only and i not in only:
            continue
        if not s.get("cam"):
            out.append(f"- **{s['name']}**: no camera")
            continue
        out.append(f"- **{s['name']}**: {s['people']} people · lights "
                   f"{'OFF' if not s['lights'] else f'{s['dim'] * 100:.0f}%'} · fan {'ON' if s['fan'] else 'OFF'} · "
                   f"AC {s['ac'].upper()} · daylight {s['daylight'] * 100:.0f}% · next {s['horizon']}s: "
                   f"{s['p_occ'] * 100:.0f}% occupied")
    return "\n".join(out) or "No live data yet."


def answer_rules(text, ctx):
    t, rooms = text.lower(), ctx["rooms"]
    idx = _rooms_in(t, rooms)
    if re.search(r"\b(which|most|worst|rank|highest)\b", t) and re.search(r"wast|energy|consum", t):
        return rank_rooms(ctx)
    if idx and re.search(r"wast|why|energy|expens|consum|inefficien|bad", t):
        return "\n\n".join(explain_room(i, ctx) for i in idx)
    if re.search(r"sav(e|ed|ing)", t):
        s = ctx["stats"]
        sv = max(0.0, s.get("base", 0) - s.get("ctrl", 0))
        return (f"Auto-control saved **{sv:.2f} kWh** so far (₹{sv * TARIFF_INR:.0f}, {sv * GRID_KG_PER_KWH:.2f} kg CO₂), "
                f"of which daylight dimming contributed {s.get('dim', 0):.2f} kWh.")
    if re.search(r"daylight|sun|dim|bright", t):
        return "Daylight and lamp levels:\n" + status_text(ctx, idx or None)
    if re.search(r"predict|forecast|next|will|expect", t):
        return "Occupancy forecast:\n" + status_text(ctx, idx or None)
    if re.search(r"status|now|current|how many|people|occup|state", t) or idx:
        return status_text(ctx, idx or None)
    return ("I can explain energy waste (*Why was Lab 201 wasteful?*), rank rooms, report savings, give the occupancy "
            "forecast, and control devices (*Switch off the Library AC*, *Dim Classroom 101 lights to 40%*, "
            "*Resume auto*).\n\n" + status_text(ctx))


# ---------------- LLM (optional) ----------------
def _context_json(ctx):
    rooms = []
    for i, r in enumerate(ctx["rooms"]):
        s = ctx["states"][i] if i < len(ctx["states"]) else {}
        rooms.append(dict(name=r.name, has_camera=bool(s.get("cam")), waste=waste_breakdown(r),
                          now={k: s.get(k) for k in ("people", "lights", "dim", "fan", "ac", "daylight", "p_occ", "p_people")}
                          if s.get("cam") else None))
    return json.dumps(dict(rooms=rooms, totals=ctx["stats"], recent_log=ctx["log"][:8]), default=str)


def _ask_llm(text, ctx):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    system = ("You are the AI copilot of a Green Building Digital Twin. Answer ONLY from the data given (energy in kWh "
              "from a short simulated run). Be concise (max 5 sentences). If the user asks you to control devices, include "
              'commands. Reply with ONLY JSON: {"answer": "...", "commands": [{"room": "<room name or all>", '
              '"device": "lights|fan|ac|dim|auto", "state": "on|off|eco|<0-100 for dim>"}]}. Use [] if no command.')
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=700, system=system,
            messages=[{"role": "user", "content": f"DATA:\n{_context_json(ctx)}\n\nUSER: {text}"}])
        raw = re.sub(r"^```(?:json)?|```$", "", msg.content[0].text.strip(), flags=re.M).strip()
        data = json.loads(raw)
    except Exception:
        return None
    cmds, rooms = [], ctx["rooms"]
    for c in data.get("commands", []) or []:
        try:
            idx = list(range(len(rooms))) if str(c["room"]).lower() == "all" else _rooms_in(str(c["room"]).lower(), rooms)
            dev, st = c["device"], c.get("state")
            if dev not in ("lights", "fan", "ac", "dim", "auto"):
                continue
            if dev == "dim":
                st = max(0.0, min(100.0, float(str(st).strip("%")))) / 100
            elif dev != "auto" and str(st).lower() not in ("on", "off", "eco"):
                continue
            for i in idx:
                cmds.append(dict(room=i, device=dev, state=str(st).lower() if dev not in ("dim", "auto") else st, seconds=300))
        except Exception:
            continue
    return str(data.get("answer", "")), cmds


def handle(text, ctx):
    """-> (reply text, list of commands to execute)"""
    rooms = ctx["rooms"]
    if not rooms:
        return "Start the live twin first - then I can answer questions and control the devices.", []
    cmds = rule_commands(text, rooms)
    if cmds == "NEED_ROOM":
        return "Which room? (" + ", ".join(r.name for r in rooms) + ", or *all*)", []
    if cmds:
        ok, notes = [], []
        for c in cmds:
            live = c["room"] < len(ctx["states"]) and ctx["states"][c["room"]].get("cam")
            if c["device"] != "auto" and not live:
                notes.append(f"⚠️ {rooms[c['room']].name} has no live camera, so I can't control it.")
            else:
                ok.append(c)
        return "\n\n".join([describe(c, rooms) for c in ok] + notes), ok
    llm = _ask_llm(text, ctx)
    if llm and llm[0]:
        ok = [c for c in llm[1] if c["room"] < len(ctx["states"]) and ctx["states"][c["room"]].get("cam")]
        return llm[0] + ("\n\n" + "\n\n".join(describe(c, rooms) for c in ok) if ok else ""), ok
    return answer_rules(text, ctx), []
