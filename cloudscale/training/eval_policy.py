"""
Evaluate a trained PPO policy and produce plots + a metrics JSON.

Usage
-----
    python -m cloudscale.training.eval_policy --model cloudscale/models/ppo_seed42.zip \\
        --episodes 5 --out-dir cloudscale/logs/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import structlog

LOG = structlog.get_logger("cloudscale.eval")

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_from_config():
    from cloudscale.envs.k8s_spot_env import K8sSpotConfig, K8sSpotEnv
    cfg = K8sSpotConfig()
    return K8sSpotEnv(config=cfg)


def rollout(model, env, seed: int) -> dict[str, list]:
    obs, _ = env.reset(seed=seed)
    history = {
        "t": [], "spot_nodes": [], "ondemand_nodes": [], "spot_price": [],
        "cpu_util": [], "mem_util": [], "p95_ms": [], "reward": [],
        "action": [], "cost_cum": [], "sla_breach": [],
    }
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        history["t"].append(env.t)
        history["spot_nodes"].append(env.spot_nodes)
        history["ondemand_nodes"].append(env.ondemand_nodes)
        history["spot_price"].append(env.spot_price)
        history["cpu_util"].append(env.last_cpu_util)
        history["mem_util"].append(env.last_mem_util)
        history["p95_ms"].append(env.last_p95_ms)
        history["action"].append(int(action))
        obs, r, term, trunc, info = env.step(int(action))
        history["reward"].append(r)
        history["cost_cum"].append(env.episode_cost)
        history["sla_breach"].append(bool(info.get("sla_breach")))
        done = term or trunc
    return history


def summarize(episode: dict[str, list]) -> dict[str, float]:
    return {
        "total_reward": float(sum(episode["reward"])),
        "total_cost_usd": float(episode["cost_cum"][-1]),
        "sla_violations": int(sum(episode["sla_breach"])),
        "mean_spot_nodes": float(np.mean(episode["spot_nodes"])),
        "mean_ondemand_nodes": float(np.mean(episode["ondemand_nodes"])),
        "spot_fraction": float(
            np.mean(episode["spot_nodes"])
            / max(1e-3, np.mean(episode["spot_nodes"]) + np.mean(episode["ondemand_nodes"]))
        ),
        "max_p95_ms": float(np.max(episode["p95_ms"])),
        "p95_p95_ms": float(np.percentile(episode["p95_ms"], 95)),
    }


def plot_episode(ep: dict[str, list], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(ep["t"], ep["spot_nodes"], label="Spot nodes", color="#1f77b4")
    ax.plot(ep["t"], ep["ondemand_nodes"], label="On-Demand nodes", color="#d62728")
    ax.set_ylabel("Node count")
    ax.legend(loc="upper right")
    ax.set_title("CloudScale K8s+Spot Policy Rollout")

    ax = axes[1]
    ax.plot(ep["t"], ep["cpu_util"], label="CPU", color="#2ca02c")
    ax.plot(ep["t"], ep["mem_util"], label="Memory", color="#ff7f0e", alpha=0.7)
    ax.axhline(0.7, color="grey", linestyle="--", alpha=0.5, label="Pressure threshold")
    ax.set_ylabel("Utilization")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(ep["t"], ep["p95_ms"], color="#9467bd")
    ax.axhline(250, color="red", linestyle="--", alpha=0.6, label="SLA (250ms)")
    ax.set_ylabel("p95 latency (ms)")
    ax.legend(loc="upper right")

    ax = axes[3]
    ax.plot(ep["t"], ep["spot_price"], color="#8c564b")
    ax.set_ylabel("Spot $/hr")
    ax.set_xlabel("sim step (1 min)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to PPO .zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "cloudscale" / "logs" / "eval")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> int:
    from stable_baselines3 import PPO
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("eval.loading", model=str(args.model))
    model = PPO.load(args.model)

    env = load_env_from_config()
    all_metrics: list[dict[str, float]] = []
    for i in range(args.episodes):
        ep = rollout(model, env, seed=args.seed + i)
        m = summarize(ep)
        all_metrics.append(m)
        LOG.info("eval.episode", idx=i, **m)
        plot_episode(ep, args.out_dir / f"episode_{i}.png")

    agg = {
        "model": str(args.model),
        "n_episodes": args.episodes,
        "mean_total_reward": float(np.mean([m["total_reward"] for m in all_metrics])),
        "mean_total_cost_usd": float(np.mean([m["total_cost_usd"] for m in all_metrics])),
        "mean_sla_violations": float(np.mean([m["sla_violations"] for m in all_metrics])),
        "mean_spot_fraction": float(np.mean([m["spot_fraction"] for m in all_metrics])),
        "episodes": all_metrics,
    }
    out_json = args.out_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    LOG.info("eval.summary_written", path=str(out_json))
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
