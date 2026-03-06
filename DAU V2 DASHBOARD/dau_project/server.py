"""
server.py — Flask backend for the DAU v2 dashboard.
"""
import os, sys, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dau_v2.main   import run
from dau_v2        import config as cfg

app = Flask(__name__, static_folder=HERE)

# ── CORS: allow all origins ──────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(HERE, "dau_v2_dashboard_connected.html")

@app.route("/run", methods=["POST", "OPTIONS"])
def run_simulation():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data    = request.get_json(force=True)
    steps   = max(10,  min(int(data.get("steps",   200)), 1000))
    seed    = int(data.get("seed",    0))
    gamma_d = max(0.1, min(float(data.get("gamma_d", 2.0)), 20.0))
    gamma_a = max(0.1, min(float(data.get("gamma_a", 2.5)), 20.0))
    gamma_u = max(0.1, min(float(data.get("gamma_u", 5.0)), 20.0))
    p_stoch = max(0.0, min(float(data.get("p_stoch", 0.2)), 0.5))

    cfg.DSG_GAMMA       = gamma_d
    cfg.ART_GAMMA       = gamma_a
    cfg.USR_GAMMA_FINAL = gamma_u
    cfg.ENV_P_STOCH     = p_stoch
    np.random.seed(seed)

    print(f"[/run] steps={steps} seed={seed} γD={gamma_d} γA={gamma_a} γU={gamma_u} p={p_stoch}")

    hist, designer, smartphone, user = run(steps=steps, seed=seed)

    def tl(arr): return [round(float(v), 5) for v in arr]
    def ag(a):   return {"qs": [round(float(v),4) for v in a.q_s],
                         "qpi":[round(float(v),4) for v in a.q_pi]}

    return jsonify({
        "steps": steps, "seed": seed,
        "efe_dsg_sel": tl(hist["efe_dsg_sel"]), "efe_art_sel": tl(hist["efe_art_sel"]),
        "efe_usr_sel": tl(hist["efe_usr_sel"]), "efe_dsg_max": tl(hist["efe_dsg_max"]),
        "efe_art_max": tl(hist["efe_art_max"]), "efe_usr_max": tl(hist["efe_usr_max"]),
        "usr_entropy": tl(hist["usr_entropy"]), "usr_maxprob": tl(hist["usr_maxprob"]),
        "usr_gamma":   tl(hist["usr_gamma"]),
        "act_dsg": hist["act_dsg"].tolist(), "act_art": hist["act_art"].tolist(),
        "act_usr": hist["act_usr"].tolist(),
        "summary": {
            "designer":   {"efe_sel_mean": round(float(np.mean(hist["efe_dsg_sel"])),4),
                           "efe_sel_std":  round(float(np.std(hist["efe_dsg_sel"])),4),
                           "efe_max_mean": round(float(np.mean(hist["efe_dsg_max"])),4),
                           "last_action":  int(hist["act_dsg"][-1])},
            "smartphone": {"efe_sel_mean": round(float(np.mean(hist["efe_art_sel"])),4),
                           "efe_sel_std":  round(float(np.std(hist["efe_art_sel"])),4),
                           "efe_max_mean": round(float(np.mean(hist["efe_art_max"])),4),
                           "last_action":  int(hist["act_art"][-1])},
            "user":       {"efe_sel_mean": round(float(np.mean(hist["efe_usr_sel"])),4),
                           "efe_sel_std":  round(float(np.std(hist["efe_usr_sel"])),4),
                           "entropy_t0":   round(float(hist["usr_entropy"][0]),4),
                           "entropy_final":round(float(hist["usr_entropy"][-1]),4),
                           "maxprob_final":round(float(hist["usr_maxprob"][-1]),4),
                           "last_action":  int(hist["act_usr"][-1])},
        },
        "final_states": {"designer": ag(designer), "smartphone": ag(smartphone), "user": ag(user)},
    })

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "dsg_states":   cfg.DSG_STATES,
        "dsg_policies": [f"{n}+{f}" for n,f in cfg.DSG_POLICIES],
        "art_states":   cfg.ART_STATES,
        "art_policies": [f"{n}+{r}" for n,r in cfg.ART_POLICIES],
        "usr_states":   cfg.USR_STATES,
        "usr_policies": cfg.USR_POLICIES,
    })

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  DAU v2 Dashboard Server")
    print("  Open http://localhost:5000 in your browser")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
