"""
PPO training pipeline for CloudScale K8s Spot environment.

Features:
  * Single-mode (train once) or Optuna HP-search mode
  * Weights & Biases (W&B) experiment tracking
  * Reproducible seeds
  * Saves best policy to models/

Run modes
---------
  # Single training run with defaults from ppo_base.yaml
  python -m cloudscale.training.train_ppo --mode single

  # Optuna sweep with 20 trials, tracking to W&B
  python -m cloudscale.training.train_ppo --mode optuna --n-trials 20

  # Smoke test on CPU: 5k timesteps, no W&B logging
  python -m cloudscale.training.train_ppo --mode single --total-timesteps 5000 \\
      --no-wandb --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml

# --- third-party (deferred to give argparse errors first) ---
# These are imported lazily inside functions to keep --help fast.


LOG = structlog.get_logger("cloudscale.train")

REPO_ROOT = Path(__file__).resolve().parents[2]  # cloudscale/training/train_ppo.py -> repo root
CONFIGS_DIR = REPO_ROOT / "cloudscale" / "configs"
MODELS_DIR = REPO_ROOT / "cloudscale" / "models"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_config_from_yaml(cfg: dict) -> Any:
    from cloudscale.envs.k8s_spot_env import K8sSpotConfig
    return K8sSpotConfig(**cfg.get("env", {}))


def make_env_fn(env_cfg: Any):
    """Return a thunk that builds the env (used by SB3 DummyEnv / SubprocVec)."""
    from cloudscale.envs.k8s_spot_env import K8sSpotEnv
    def _thunk():
        return K8sSpotEnv(config=env_cfg)
    return _thunk


def evaluate_policy(model, env_fn, n_episodes: int = 5) -> dict[str, float]:
    """Roll out a trained policy and return aggregate metrics."""
    from cloudscale.envs.k8s_spot_env import K8sSpotEnv
    env = env_fn()
    total_rewards, total_costs, total_sla_violations = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_r, ep_c, ep_sla = 0.0, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r
            ep_c += info.get("cost", 0.0)
            if info.get("sla_breach"):
                ep_sla += 1
            done = term or trunc
        total_rewards.append(ep_r)
        total_costs.append(ep_c)
        total_sla_violations.append(ep_sla)
    return {
        "eval/mean_reward": float(np.mean(total_rewards)),
        "eval/mean_cost": float(np.mean(total_costs)),
        "eval/mean_sla_violations": float(np.mean(total_sla_violations)),
        "eval/reward_std": float(np.std(total_rewards)),
    }


# ---------------------------------------------------------------------------
# Weights & Biases setup
# ---------------------------------------------------------------------------

def setup_wandb(cfg: dict, use_wandb: bool, seed: int) -> bool:
    """Initialise a W&B run. Returns True if W&B was set up successfully."""
    if not use_wandb:
        LOG.info("wandb.disabled")
        return False

    import wandb

    wandb_cfg = cfg.get("wandb", {})

    # Read token from environment (set via Colab Secrets → WANDB_API_KEY)
    token = os.environ.get("WANDB_API_KEY", "").strip()
    if not token:
        LOG.error(
            "wandb.no_token",
            hint="Set WANDB_API_KEY in Colab Secrets, or pass --no-wandb for local smoke tests.",
        )
        raise RuntimeError("WANDB_API_KEY is required before training can start.")

    try:
        wandb.login(key=token, relogin=True)
        wandb.init(
            entity=wandb_cfg.get("entity", "bhanujbhalla7-mpstme"),
            project=wandb_cfg.get("project", "cloudscale-ppo-phase1"),
            config=cfg,
            name=f"ppo-seed{seed}-{int(time.time())}",
            tags=wandb_cfg.get("tags", ["ppo", "phase1", "k8s-spot"]),
            save_code=True,
        )
        LOG.info("wandb.connected", entity=wandb_cfg.get("entity"), project=wandb_cfg.get("project"))
        return True
    except Exception as e:
        LOG.error("wandb.connection_failed", error=str(e))
        raise RuntimeError(
            f"W&B connection failed: {e}\n"
            "Verify your WANDB_API_KEY is correct (https://wandb.ai/authorize)."
        ) from e


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(args, cfg: dict) -> dict[str, float]:
    """Train PPO once with the (possibly Optuna-overridden) config."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    set_global_seed(args.seed)
    env_cfg = env_config_from_yaml(cfg)
    env_cfg.seed = args.seed

    # Wrap in SB3 Monitor for episodic stats
    def _make_monitored():
        from cloudscale.envs.k8s_spot_env import K8sSpotEnv
        return Monitor(K8sSpotEnv(config=env_cfg))
    env = DummyVecEnv([_make_monitored])

    ppo_kwargs = dict(cfg.get("ppo", {}))
    # Resolve activation_fn string -> callable
    act_str = ppo_kwargs.get("policy_kwargs", {}).pop("activation_fn", "tanh")
    import torch.nn as nn
    ppo_kwargs["policy_kwargs"]["activation_fn"] = (
        nn.Tanh if act_str == "tanh" else nn.ReLU
    )
    ppo_kwargs["device"] = args.device

    LOG.info("training.start", timesteps=args.total_timesteps, device=args.device)
    t0 = time.time()
    model = PPO(env=env, **ppo_kwargs)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    train_seconds = time.time() - t0
    LOG.info("training.done", seconds=round(train_seconds, 1))

    # Evaluate
    metrics = evaluate_policy(model, _make_monitored, n_episodes=args.eval_episodes)
    metrics["train/seconds"] = train_seconds
    metrics["train/total_timesteps"] = args.total_timesteps
    LOG.info("eval.results", **metrics)

    # Save model
    model_path = MODELS_DIR / f"ppo_seed{args.seed}.zip"
    model.save(model_path)
    LOG.info("model.saved", path=str(model_path))

    # W&B logging
    if args.use_wandb:
        import wandb
        # Log hyperparameters
        flat_params = {k: v for k, v in ppo_kwargs.items() if not isinstance(v, dict)}
        flat_params.update(asdict(env_cfg))
        wandb.config.update(flat_params, allow_val_change=True)
        # Log metrics
        wandb.log(metrics)
        # Upload model artifact
        wandb_cfg = cfg.get("wandb", {})
        if wandb_cfg.get("log_model_artifact", True):
            artifact = wandb.Artifact(
                name=f"ppo-model-seed{args.seed}",
                type="model",
                description="Trained PPO policy for K8s Spot orchestration",
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            LOG.info("wandb.artifact_logged", path=str(model_path))

    return metrics


# ---------------------------------------------------------------------------
# Optuna sweep
# ---------------------------------------------------------------------------

def train_optuna(args, cfg: dict) -> dict[str, Any]:
    import optuna
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import torch.nn as nn

    optuna_cfg = cfg.get("optuna", {})
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=optuna_cfg.get("study_name", "cloudscale-ppo-v1"),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    LOG.info("optuna.start", n_trials=args.n_trials)

    def objective(trial: optuna.Trial) -> float:
        # Sample HP overrides
        trial_cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
        ppo = trial_cfg.setdefault("ppo", {})
        ppo["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ppo["n_steps"] = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        ppo["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
        ppo["n_epochs"] = trial.suggest_categorical("n_epochs", [5, 10, 20])
        ppo["gamma"] = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
        ppo["gae_lambda"] = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.99])
        ppo["clip_range"] = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        ppo["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.01, 0.05])

        env_cfg = env_config_from_yaml(trial_cfg)
        env_cfg.seed = args.seed

        def _make_monitored():
            from cloudscale.envs.k8s_spot_env import K8sSpotEnv
            return Monitor(K8sSpotEnv(config=env_cfg))
        env = DummyVecEnv([_make_monitored])

        ppo_kwargs = dict(trial_cfg.get("ppo", {}))
        act_str = ppo_kwargs.get("policy_kwargs", {}).pop("activation_fn", "tanh")
        ppo_kwargs["policy_kwargs"]["activation_fn"] = (
            nn.Tanh if act_str == "tanh" else nn.ReLU
        )
        ppo_kwargs["device"] = args.device
        ppo_kwargs["verbose"] = 0

        model = PPO(env=env, **ppo_kwargs)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        metrics = evaluate_policy(model, _make_monitored, n_episodes=args.eval_episodes)
        mean_r = metrics["eval/mean_reward"]

        if args.use_wandb:
            import wandb
            wandb.log({
                "trial/number": trial.number,
                **trial.params,
                **metrics,
            })
        # Optuna pruning hook (intermediate)
        trial.report(mean_r, step=args.total_timesteps)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return mean_r

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    LOG.info(
        "optuna.done",
        seconds=round(time.time() - t0, 1),
        best_value=study.best_value,
        best_params=study.best_params,
    )
    if args.use_wandb:
        import wandb
        wandb.log({
            "optuna/best_value": study.best_value,
            **{f"optuna/best_{k}": v for k, v in study.best_params.items()},
        })
    return {"best_value": study.best_value, "best_params": study.best_params}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CloudScale PPO trainer")
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "ppo_base.yaml")
    p.add_argument("--mode", choices=["single", "optuna"], default="single")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.set_defaults(use_wandb=True)
    return p.parse_args()


def main() -> int:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    args = parse_args()
    cfg = load_yaml(args.config)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    setup_wandb(cfg, use_wandb=args.use_wandb, seed=args.seed)

    if args.mode == "single":
        result = train_single(args, cfg)
    else:
        result = train_optuna(args, cfg)

    # Finish W&B run cleanly
    if args.use_wandb:
        import wandb
        wandb.finish()

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
