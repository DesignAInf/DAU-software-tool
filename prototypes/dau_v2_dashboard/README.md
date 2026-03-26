# DAU v2 dashboard

Interactive dashboard: Flask backend runs the real `dau_v2` simulation; the browser renders charts on HTML canvas (no static AI-generated screenshots).

## Dependencies

```bash
cd dau_project
pip install -r requirements.txt
```

## Run the dashboard

```bash
cd dau_project
python server.py
```

Open `http://127.0.0.1:5000`, adjust sliders, then **Run Simulation**.

## Static matplotlib figures (optional)

To export PNGs of the same metrics shown in the UI (EFE, user entropy, γ):

```bash
cd dau_project
python generate_reference_plots.py
```

Outputs appear in `dau_project/results/`.

## Full documentation

See `README_DASHBOARD.md` for architecture, API, and Active Inference notes.
