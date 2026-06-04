# Phase 1 — The Simulator

This is where the "brain" gets built. Everything in this phase runs
locally on your Mac (CPU is fine) or on a free Google Colab T4 GPU.

---

## What you get out of Phase 1

1. A custom **Gymnasium environment** (`cloudscale/envs/k8s_spot_env.py`)
   that mimics a K8s cluster with mixed Spot/On-Demand nodes, time-varying
   load, random spot interruptions, and an SLA on p95 latency.
2. A **PPO training pipeline** (`cloudscale/training/train_ppo.py`) with
   both a single-run mode and an **Optuna hyperparameter sweep**.
3. **MLflow tracking to DagsHub** so every experiment (params, metrics,
   model artifact) is automatically logged.
4. An **eval harness** (`cloudscale/training/eval_policy.py`) that
   rolls out the trained policy and produces nice plots.
5. A **Google Colab notebook** that runs the whole training in ~5 minutes
   on a free T4 GPU.

---

## Files

| File | Purpose |
|---|---|
| `cloudscale/envs/k8s_spot_env.py` | The Gym env (10-dim obs, 6 discrete actions) |
| `cloudscale/configs/env_base.yaml` | Env hyperparameters (load dynamics, pricing, SLA) |
| `cloudscale/configs/ppo_base.yaml` | PPO + Optuna defaults |
| `cloudscale/training/train_ppo.py` | `python -m cloudscale.training.train_ppo ...` |
| `cloudscale/training/eval_policy.py` | `python -m cloudscale.training.eval_policy ...` |
| `cloudscale/notebooks/train_colab.ipynb` | Colab launcher |

---

## How to run

### A. Local quick test (CPU, ~2 seconds)

```bash
source cloudscale-venv/bin/activate
python -m cloudscale.training.train_ppo \
    --mode single --total-timesteps 2000 --no-mlflow --device cpu
```

### B. Local Optuna sweep (CPU, ~10-20 min for 5 trials × 20k steps)

```bash
python -m cloudscale.training.train_ppo \
    --mode optuna --n-trials 5 --total-timesteps 20000 --no-mlflow --device cpu
```

### C. Colab T4 training (recommended, 5-20 min)

See `cloudscale/notebooks/train_colab.ipynb`. You'll need:

- **DagsHub user token** (free, https://dagshub.com/user/settings/tokens)
- Add it as `DAGSHUB_USER_TOKEN` in Colab's secrets panel (🔑 icon)
- Then run all cells in order

### D. Evaluate

```bash
python -m cloudscale.training.eval_policy \
    --model cloudscale/models/ppo_seed42.zip \
    --episodes 5 --out-dir cloudscale/logs/eval
```

Plots land in `cloudscale/logs/eval/episode_*.png` and a summary in
`summary.json`.

---

## What to look for

| Metric | What it tells you | Target |
|---|---|---|
| `eval/mean_reward` | Total reward across an episode — higher is better | ≥ +5 after 200k steps |
| `eval/mean_cost` | USD spent per episode | Should drop as agent learns spot |
| `eval/mean_sla_violations` | Steps where p95 > 250ms | Should approach 0 |
| `eval/mean_spot_fraction` | What % of nodes are spot | Should climb toward 0.6-0.8 |
| `max_p95_ms` | Worst-case latency seen | Should stay near 250ms |

If `mean_spot_fraction` is near 0 after 200k steps, the policy hasn't
learned the cost trade-off. Try:
- More `total_timesteps` (200k → 500k)
- Lower `churn_penalty` (agent is being too cautious)
- Increase `ent_coef` in Optuna (more exploration)

---

## Next: Phase 2

Once you have a trained `ppo_seed42.zip` that scores well on the metrics
above, we move to building the **real dataplane**: a Civo K3s cluster
with Prometheus, Kafka, Feast, and an Ngrok tunnel back to your local
FastAPI.
