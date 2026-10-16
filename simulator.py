"""Synthetic occupancy so the demo works without a camera."""
import math
import random


def demo_tick(room, tick, total, ac_always=True):
    offset = (sum(map(ord, room.name)) % 10) / 10
    wave = max(0.0, math.sin((tick / total * 2 + offset) * math.pi))
    people = int(room.capacity * wave * random.uniform(0.4, 0.9))
    if people < 3:
        people = 0
    if people > 0:
        room.lights_on = True
    elif random.random() < 0.05:        # lights are often forgotten
        room.lights_on = False
    ac_on = True if ac_always else people > 0
    return people, room.lights_on, ac_on
