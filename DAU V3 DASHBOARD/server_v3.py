"""
server.py — Flask backend for the DAU v3 dashboard.

Endpoints:
  GET  /           — dashboard HTML
  GET  /config     — agent labels
  POST /run        — single run, all 4 profiles
  POST /analysis   — batch run (n_seeds), returns aggregated stats
"""

import os, sys, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dau_v3.main     import run
from dau_v3.analysis import run_batch, compute_scalars
from dau_v3          import config as cfg

app = Flask(__name__, static_folder=HERE)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(HERE, "dau_v3_dashboard.html")


@app.route("/config")
def get_config():
    return jsonify({
        "dsg_states":   cfg.DSG_STATES,
        "dsg_policies": [f"{n}+{f}" for n, f in cfg.DSG_POLICIES],
        "art_states":   cfg.ART_STATES,
        "art_policies": [f"{n}+{r}" for n, r in cfg.ART_POLICIES],
        "usr_states":   cfg.USR_STATES,
        "usr_policies": cfg.USR_POLICIES,
        "profiles":     {
            p: {
                "desc":          cfg.USER_PROFILES[p]["desc"],
                "vulnerability": cfg.USER_PROFILES[p]["vulnerability"],
                "gamma_init":    cfg.USER_PROFILES[p]["gamma_init"],
                "gamma_final":   cfg.USER_PROFILES[p]["gamma_final"],
            }
            for p in cfg.USER_PROFILES
        },
    })


@app.route("/run", methods=["POST", "OPTIONS"])
def run_single():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data    = request.get_json(force=True)
    steps   = max(50, min(int(data.get("steps", 300)), 1000))
    seed    = int(data.get("seed", 0))
    p_stoch = max(0.0, min(float(data.get("p_stoch", cfg.ENV_P_STOCH)), 0.5))
    gamma_d = max(0.1, min(float(data.get("gamma_d", cfg.DSG_GAMMA)), 20.0))
    gamma_a = max(0.1, min(float(data.get("gamma_a", cfg.ART_GAMMA)), 20.0))

    cfg.ENV_P_STOCH = p_stoch
    cfg.DSG_GAMMA   = gamma_d
    cfg.ART_GAMMA   = gamma_a
    np.random.seed(seed)

    print(f"[/run] steps={steps} seed={seed} p={p_stoch} γD={gamma_d} γA={gamma_a}")

    hist_all, designer, smartphone, users = run(steps=steps, seed=seed)

    def tl(arr): return [round(float(v), 5) for v in arr]
    def ti(arr): return [int(v) for v in arr]

    profiles = list(cfg.USER_PROFILES.keys())
    out = {
        "steps": steps, "seed": seed,
        "shared": {
            "efe_dsg_sel": tl(hist_all[profiles[0]]["efe_dsg_sel"]),
            "efe_art_sel": tl(hist_all[profiles[0]]["efe_art_sel"]),
            "act_dsg":     ti(hist_all[profiles[0]]["act_dsg"]),
            "act_art":     ti(hist_all[profiles[0]]["act_art"]),
            "designer_summary": {
                "efe_mean": round(float(hist_all[profiles[0]]["efe_dsg_sel"].mean()), 4),
                "efe_std":  round(float(hist_all[profiles[0]]["efe_dsg_sel"].std()),  4),
                "last_action": int(hist_all[profiles[0]]["act_dsg"][-1]),
            },
            "smartphone_summary": {
                "efe_mean": round(float(hist_all[profiles[0]]["efe_art_sel"].mean()), 4),
                "efe_std":  round(float(hist_all[profiles[0]]["efe_art_sel"].std()),  4),
                "last_action": int(hist_all[profiles[0]]["act_art"][-1]),
            },
            "final_dsg": {
                "qs":  [round(float(v), 4) for v in designer.q_s],
                "qpi": [round(float(v), 4) for v in designer.q_pi],
            },
            "final_art": {
                "qs":  [round(float(v), 4) for v in smartphone.q_s],
                "qpi": [round(float(v), 4) for v in smartphone.q_pi],
            },
        },
        "profiles": {},
    }

    for p in profiles:
        h  = hist_all[p]
        u  = users[p]
        out["profiles"][p] = {
            "efe_usr_sel": tl(h["efe_usr_sel"]),
            "usr_entropy": tl(h["usr_entropy"]),
            "usr_maxprob": tl(h["usr_maxprob"]),
            "usr_gamma":   tl(h["usr_gamma"]),
            "user_state":  ti(h["user_state"]),
            "act_usr":     ti(h["act_usr"]),
            "summary": {
                "efe_mean":      round(float(h["efe_usr_sel"].mean()), 4),
                "efe_std":       round(float(h["efe_usr_sel"].std()),  4),
                "entropy_t0":    round(float(h["usr_entropy"][0]),     4),
                "entropy_final": round(float(h["usr_entropy"][-1]),    4),
                "maxprob_final": round(float(h["usr_maxprob"][-1]),    4),
                "last_action":   int(h["act_usr"][-1]),
                "final_state":   int(h["user_state"][-1]),
                "churn": bool(int(h["user_state"][-1]) in [8, 9]),
            },
            "final_state": {
                "qs":  [round(float(v), 4) for v in u.q_s],
                "qpi": [round(float(v), 4) for v in u.q_pi],
            },
        }

    return jsonify(out)


@app.route("/analysis", methods=["POST", "OPTIONS"])
def run_analysis():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data    = request.get_json(force=True)
    steps   = max(50,  min(int(data.get("steps",   300)), 500))
    n_seeds = max(5,   min(int(data.get("n_seeds", 20)),  100))
    p_stoch = max(0.0, min(float(data.get("p_stoch", cfg.ENV_P_STOCH)), 0.5))
    gamma_d = max(0.1, min(float(data.get("gamma_d", cfg.DSG_GAMMA)), 20.0))
    gamma_a = max(0.1, min(float(data.get("gamma_a", cfg.ART_GAMMA)), 20.0))

    cfg.ENV_P_STOCH = p_stoch
    cfg.DSG_GAMMA   = gamma_d
    cfg.ART_GAMMA   = gamma_a

    print(f"[/analysis] steps={steps} n_seeds={n_seeds} p={p_stoch} γD={gamma_d} γA={gamma_a}")

    agg     = run_batch(steps=steps, n_seeds=n_seeds)
    scalars = compute_scalars(agg, steps)
    profiles = list(cfg.USER_PROFILES.keys())

    # Mean timeseries (averaged over seeds)
    out = {"steps": steps, "n_seeds": n_seeds, "profiles": {}}
    for p in profiles:
        sc = scalars[p]
        out["profiles"][p] = {
            "efe_mean":    [round(float(v), 5) for v in agg[p]["efe_usr_sel"].mean(axis=0)],
            "efe_std":     [round(float(v), 5) for v in agg[p]["efe_usr_sel"].std(axis=0)],
            "entropy_mean":[round(float(v), 5) for v in agg[p]["usr_entropy"].mean(axis=0)],
            "entropy_std": [round(float(v), 5) for v in agg[p]["usr_entropy"].std(axis=0)],
            "state_dist":  [round(float(np.mean(agg[p]["user_state"] == s)), 4)
                            for s in range(cfg.USR_N_STATES)],
            "policy_dist": [round(float(np.mean(agg[p]["act_usr"] == pi)), 4)
                            for pi in range(cfg.USR_N_POLICIES)],
            "scalars": {
                "efe_mean":     round(sc["efe_mean"],     4),
                "efe_std":      round(sc["efe_std"],      4),
                "churn_rate":   round(sc["churn_rate"],   4),
                "stress_idx":   round(sc["stress_idx"],   4),
                "entropy_drop": round(sc["entropy_drop"], 4),
                "resist_t_med": int(np.median(sc["resist_t"])),
            },
        }
    return jsonify(out)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DAU v3 Dashboard Server")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)
