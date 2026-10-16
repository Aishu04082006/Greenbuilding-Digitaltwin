# Green Building Digital Twin

## Run (VS Code terminal)
    python -m venv venv
    venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt
    streamlit run app.py

Optional GenAI: `setx ANTHROPIC_API_KEY "your_key"` (restart VS Code). Works offline without it.

## Files
- app.py        Streamlit dashboard (floor-plan heatmap, KPIs, what-if, AI, CSV export)
- twin.py       Digital twin: rooms, sensors, energy/CO2/cost, alerts, what-if
- vision.py     YOLOv8 people counting + lights detection
- simulator.py  Synthetic occupancy for demo mode
- ai.py         GenAI recommendations + offline fallback
