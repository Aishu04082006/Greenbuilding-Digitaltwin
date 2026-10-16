"""GenAI recommendations with an offline rule-based fallback."""
import os
from twin import summarize, CO2_LIMIT_PPM, GRID_KG_PER_KWH


def _llm(prompt):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=600,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception:
        return None


def _context(rooms):
    lines = [f"Building totals: {summarize(rooms)}"]
    for r in rooms:
        lines.append(f"{r.name}: {summarize([r])}; latest reading: {r.history[-1] if r.history else {}}")
    return "\n".join(lines)


def rule_based(rooms):
    tips = []
    for r in rooms:
        h = r.history
        if not h:
            continue
        lights = sum(1 for x in h if x["people"] == 0 and x["lights_on"])
        ac = sum(1 for x in h if x["people"] == 0 and x["ac_on"])
        co2 = sum(1 for x in h if x["co2"] > CO2_LIMIT_PPM)
        if lights:
            tips.append(f"- **{r.name}**: lights were on while empty for {lights} readings. Fit occupancy-based auto-off.")
        if ac:
            tips.append(f"- **{r.name}**: AC ran while empty for {ac} readings. Link AC to the timetable/occupancy.")
        if co2:
            tips.append(f"- **{r.name}**: CO₂ exceeded {CO2_LIMIT_PPM} ppm {co2} times. Add fresh-air ventilation.")
    s = summarize(rooms)
    tips.append(f"- Avoidable waste: **{s['waste_kwh']} kWh** (~{s['waste_kwh'] * GRID_KG_PER_KWH:.1f} kg CO₂).")
    return "\n".join(tips) or "No major waste detected."


def recommend(rooms):
    prompt = ("You are an energy-efficiency expert for a green building digital twin. "
              "Using this data, give 5 specific, prioritised recommendations to cut energy waste, "
              "carbon emissions and improve indoor sustainability. Be concise.\n\n" + _context(rooms))
    return _llm(prompt) or rule_based(rooms)


def ask(question, rooms):
    prompt = ("Answer the question using only this building data. Be brief.\n\n"
              f"{_context(rooms)}\n\nQuestion: {question}")
    return _llm(prompt) or ("(Offline mode - set ANTHROPIC_API_KEY for chat.)\n\n" + rule_based(rooms))
