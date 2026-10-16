"""Digital twin model: rooms, simulated sensors, energy / carbon maths, what-if."""
import random
from dataclasses import dataclass, field

GRID_KG_PER_KWH = 0.82   # approx. India grid emission factor (edit if needed)
TARIFF_INR = 8.0         # Rs per kWh (edit for your campus)
CO2_LIMIT_PPM = 1000     # indoor air-quality limit
TEMP_MAX_C = 27.0        # comfort limit


@dataclass
class Room:
    name: str
    capacity: int = 40
    lights_kw: float = 0.4
    ac_kw: float = 1.5
    equipment_kw: float = 0.3
    fan_kw: float = 0.075
    lights_on: bool = False
    fan_on: bool = False
    ac_on: bool = True
    history: list = field(default_factory=list)

    def sensors(self, people):
        """Simulated IoT sensors (replace with real MQTT/serial data if available)."""
        temp = 24 + 0.08 * people + (0 if self.ac_on else 3) + random.uniform(-0.4, 0.4)
        co2 = 420 + 30 * people + random.uniform(-25, 25)
        lux = (320 if self.lights_on else 90) + random.uniform(-20, 20)
        return round(temp, 1), int(co2), int(lux)

    def update(self, people, lights_on, ac_on=True, interval_min=10, fan_on=None):
        if fan_on is None:
            fan_on = lights_on
        self.lights_on, self.ac_on, self.fan_on = lights_on, ac_on, fan_on
        hours = interval_min / 60
        occupied = people > 0
        temp, co2, lux = self.sensors(people)

        # Actual (as-is) load
        actual = ((self.lights_kw if lights_on else 0)
                  + (self.ac_kw if ac_on else 0)
                  + (self.fan_kw if fan_on else 0)
                  + self.equipment_kw * (1.0 if occupied else 0.5))

        # Optimised load the twin recommends
        if occupied:
            load_factor = 0.6 + 0.4 * min(1.0, people / (0.5 * self.capacity))
            optimal = self.lights_kw + self.ac_kw * load_factor + self.equipment_kw + self.fan_kw
        else:
            optimal = self.equipment_kw * 0.05      # smart plugs, all else off

        rec = dict(
            room=self.name, tick=len(self.history), hours=hours, people=people,
            lights_on=lights_on, ac_on=ac_on, fan_on=fan_on, temp=temp, co2=co2, lux=lux,
            actual_kw=round(actual, 3), optimal_kw=round(optimal, 3),
            waste_kw=round(max(0.0, actual - optimal), 3),
        )
        self.history.append(rec)
        return rec


def summarize(rooms, tariff=TARIFF_INR):
    kwh = sum(r["actual_kw"] * r["hours"] for rm in rooms for r in rm.history)
    opt = sum(r["optimal_kw"] * r["hours"] for rm in rooms for r in rm.history)
    waste = max(0.0, kwh - opt)
    return dict(
        kwh=round(kwh, 2), optimal_kwh=round(opt, 2), waste_kwh=round(waste, 2),
        co2_kg=round(kwh * GRID_KG_PER_KWH, 2), cost_inr=round(kwh * tariff),
        score=round(100 * (1 - waste / kwh)) if kwh else 100,
    )


def alerts_for(room):
    if not room.history:
        return []
    r, out = room.history[-1], []
    if r["people"] == 0 and r["lights_on"]:
        out.append(f"💡 {room.name}: lights ON in empty room")
    if r["people"] == 0 and r["ac_on"]:
        out.append(f"❄️ {room.name}: AC running in empty room")
    if r["people"] == 0 and r.get("fan_on"):
        out.append(f"🌀 {room.name}: fan running in empty room")
    if r["co2"] > CO2_LIMIT_PPM:
        out.append(f"🫁 {room.name}: CO₂ {r['co2']} ppm - ventilate")
    if r["temp"] > TEMP_MAX_C and r["people"] > 0:
        out.append(f"🌡️ {room.name}: {r['temp']}°C is above comfort level")
    return out


def what_if(rooms, ac_off_empty=True, lights_off_empty=True,
            plugs_off_empty=False, setpoint_raise_c=0, fans_off_empty=True):
    """kWh saved over the recorded period if the chosen policies were applied."""
    saved = 0.0
    for rm in rooms:
        for r in rm.history:
            h = r["hours"]
            if r["people"] == 0:
                if ac_off_empty and r["ac_on"]:
                    saved += rm.ac_kw * h
                if lights_off_empty and r["lights_on"]:
                    saved += rm.lights_kw * h
                if fans_off_empty and r.get("fan_on"):
                    saved += rm.fan_kw * h
                if plugs_off_empty:
                    saved += rm.equipment_kw * 0.45 * h
            elif r["ac_on"]:
                saved += rm.ac_kw * h * 0.06 * setpoint_raise_c   # ~6% per degC
    return saved