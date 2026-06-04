# CloudScale AI — Autonomous FinOps Orchestrator

> **A multi-cloud AI orchestrator that continuously ingests real-time
> Kubernetes telemetry, anticipates workload spikes, and autonomously
> shifts traffic between Spot and On-Demand instances using
> Reinforcement Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-orange)](https://gymnasium.farama.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-green)](https://stable-baselines3.readthedocs.io)

---

## Why this exists

Cloud bills are 30-40% higher than they need to be. The waste lives in
idle compute, oversized nodes, and over-reliance on On-Demand when Spot
would do. Today, FinOps is mostly a human running a spreadsheet once a
month. We want it to be a self-driving control loop that:

1. **Watches** Prometheus metrics from a Kubernetes cluster in real time
2. **Decides** when to scale, and whether to use Spot or On-Demand
3. **Acts** through `kubectl` with deterministic OPA safety guardrails
4. **Learns** from every decision, getting sharper over time

The decision-making is a **PPO Reinforcement Learning agent** (Stable-Baselines3),
trained on a custom Gymnasium simulator and served locally for sub-millisecond
inference.

---

## Architecture — the three planes

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│      LOCAL PLANE         │    │      CLOUD PLANE         │    │       DATAPLANE          │
│   (Control & Inference)  │    │   (Training & MLOps)     │    │ (Real infra)             │
│                          │    │                          │    │                          │
│  FastAPI backend         │◄──►│  Google Colab (T4 GPU)   │    │  Civo K3s cluster        │
│  LangChain agent         │    │  DagsHub MLflow          │    │  Prometheus              │
│  Trained PPO inference   │    │  Optuna HP search        │    │  Apache Kafka            │
│  OrbStack (build only)   │    │                          │    │  Feast (feature store)   │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                                                       ▲
                                                                       │  Ngrok tunnel
                                                                       ▼
                                                              Secure local agent
```

| Plane | Lives in | Tech |
|---|---|---|
| **Local** | Your Mac | FastAPI, LangChain, Stable-Baselines3, OrbStack |
| **Cloud** | Free cloud services | Google Colab, DagsHub, Optuna |
| **Dataplane** | Civo K3s (real cluster) | K3s, Prometheus, Kafka, Feast, OPA |

---

## Phased roadmap

| Phase | Status | Deliverable |
|---|---|---|
| **1. Simulator** | ✅ Done | Custom Gym env + PPO pipeline + Optuna + MLflow + Colab notebook |
| **2. Dataplane** | ⏳ Next | Civo K3s + Prometheus + Kafka + Feast + Ngrok tunnel |
| **3. Closed Loop** | ⏳ Pending | FastAPI orchestrator + OPA guardrails + live kubectl execution |
| **4. Interface & Polish** | ⏳ Pending | LangChain NL ops + God-tier README + architecture diagrams |

---

## Quick start

### 1. Clone & setup

```bash
git clone https://github.com/bhanujjj/CloudScale-AI-Autonomous-FinOps-Orchestrator.git
cd CloudScale-AI-Autonomous-FinOps-Orchestrator

python3 -m venv cloudscale-venv
source cloudscale-venv/bin/activate
pip install -r cloudscale/configs/requirements.txt
```

### 2. Smoke-test the simulator

```bash
python cloudscale/envs/k8s_spot_env.py
```

You should see 20 random rollout steps with spot/od node counts, prices,
and latencies.

### 3. Quick local PPO training (2k steps, ~2 sec on CPU)

```bash
python -m cloudscale.training.train_ppo \
    --mode single --total-timesteps 2000 --no-mlflow --device cpu
```

### 4. Real training on Colab (200k steps, ~5 min on T4)

See [`cloudscale/notebooks/train_colab.ipynb`](cloudscale/notebooks/train_colab.ipynb).
You'll need a free DagsHub token — see [`MANUAL_STEPS.md`](./MANUAL_STEPS.md).

### 5. Evaluate a trained model

```bash
python -m cloudscale.training.eval_policy \
    --model cloudscale/models/ppo_seed42.zip \
    --episodes 5 --out-dir cloudscale/logs/eval
```

Plots rollouts into `cloudscale/logs/eval/episode_*.png`.

---

## Project layout

```
.
├── cloudscale/
│   ├── envs/
│   │   └── k8s_spot_env.py          # Custom Gymnasium env (the "brain playground")
│   ├── training/
│   │   ├── train_ppo.py             # Single + Optuna training, MLflow to DagsHub
│   │   └── eval_policy.py           # Rollout + plots + metrics JSON
│   ├── notebooks/
│   │   └── train_colab.ipynb        # Google Colab launcher
│   ├── configs/
│   │   ├── requirements.txt
│   │   ├── env_base.yaml            # Gym env hyperparameters
│   │   └── ppo_base.yaml            # PPO + Optuna defaults
│   ├── src/cloudscale/              # (Phase 3) FastAPI orchestrator goes here
│   ├── infra/                       # (Phase 2) K8s/Helm/Kafka/Feast/Prometheus/OPA
│   ├── models/                      # Trained PPO checkpoints (.zip)
│   ├── logs/                        # Eval plots + run logs
│   └── data/                        # Replay buffers, sample traces
├── MANUAL_STEPS.md                  # What YOU have to do (cloud signups, tokens)
├── .gitignore
├── LICENSE
└── README.md
```

---

## The 2026/27 "S-Tier" tech stack

| Component | Tech | Why |
|---|---|---|
| Core brain | Stable-Baselines3 (PPO) | Proves multi-objective RL mastery |
| Env | Gymnasium | Industry standard RL interface |
| HP search | Optuna | TPE + pruning, plays nice with MLflow |
| Tracking | MLflow on DagsHub | Free hosted MLOps, no infra to manage |
| Stream | Apache Kafka | Enterprise event bus, not toy CSV |
| Features | Feast | Sub-ms online feature serving |
| Telemetry | Prometheus | Native K8s monitoring |
| Guardrails | Open Policy Agent | Deterministic safety bounds |
| NL ops | LangChain + LLM | Engineers prompt the infra |
| Local runtime | OrbStack | Lighter than Docker Desktop on M-series |

---

## MLflow / DagsHub dashboard

Live experiments: <https://dagshub.com/bhanujbhalla7/CloudScale-AI-Autonomous-FinOps-Orchestrator.mlflow>

---

## License

MIT © 2026 Bhanuj Bhalla. See [LICENSE](LICENSE).
