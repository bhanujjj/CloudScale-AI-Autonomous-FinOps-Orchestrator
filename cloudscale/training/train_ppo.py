"""
PPO training pipeline for CloudScale K8s Spot environment.

Features:
  * Single-mode (train once) or Optuna HP-search mode
  * MLflow logging to DagsHub (or local fallback if DagsHub is unreachable)
  * Reproducible seeds
  * Saves best policy to models/

Run modes
---------
  # Single training run with defaults from ppo_base.yaml
  python -m cloudscale.training.train_ppo --mode single

  # Optuna sweep with 20 trials, tracking to DagsHub
  python -m cloudscale.training.train_ppo --mode optuna --n-trials 20

  # Smoke test on CPU: 5k timesteps, 1 trial, local MLflow
  python -m cloudscale.training.train_ppo --mode single --total-timesteps 5000 \\
      --no-mlflow --device cpu
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
# MLflow setup (DagsHub required unless --no-mlflow is passed)
# ---------------------------------------------------------------------------

def setup_mlflow(cfg: dict, use_mlflow: bool) -> bool:
    """Returns True if MLflow was set up successfully."""
    if not use_mlflow:
        LOG.info("mlflow.disabled")
        return False

    import mlflow
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")

    if tracking_uri and "dagshub.com" in tracking_uri:
        # Initialize DagsHub MLflow integration.
        raw_token = os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get("DAGSHUB_TOKEN") or ""
        token = "".join(raw_token.split())
        if token:
            os.environ["DAGSHUB_USER_TOKEN"] = token
            os.environ["DAGSHUB_TOKEN"] = token
        if not token:
            LOG.error(
                "dagshub.no_token",
                hint="Set DAGSHUB_USER_TOKEN in Colab Secrets, or pass --no-mlflow for local smoke tests.",
            )
            raise RuntimeError("DagsHub token is required before training can start.")
        try:
            # ── Bulletproof DagsHub MLflow auth ──────────────────────────
            # The `dagshub` Python library's token-validation path is broken
            # in Google Colab (JSONDecodeError / "token not valid" depending
            # on the code path).  We don't need the library at all.
            #
            # MLflow's REST client natively supports HTTP Basic Auth when
            # credentials are embedded in the tracking URI:
            #   https://user:token@dagshub.com/owner/repo.mlflow
            #
            # This is the same mechanism the DagsHub web UI tells you to
            # use ("set MLFLOW_TRACKING_USERNAME / PASSWORD"), but embedding
            # them in the URI guarantees they are picked up regardless of
            # import order or env-var timing issues.
            # ─────────────────────────────────────────────────────────────
            from urllib.parse import urlparse, urlunparse

            owner = mlflow_cfg.get("dagshub_repo_owner", "bhanujbhalla7")
            parsed = urlparse(tracking_uri)

            # Build an authenticated URI:  https://owner:token@dagshub.com/…
            authed_netloc = f"{owner}:{token}@{parsed.hostname}"
            if parsed.port:
                authed_netloc += f":{parsed.port}"
            authed_uri = urlunparse(parsed._replace(netloc=authed_netloc))

            # Also set the env vars as a belt-and-suspenders fallback
            os.environ["MLFLOW_TRACKING_USERNAME"] = owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token

            mlflow.set_tracking_uri(authed_uri)
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "cloudscale-ppo-phase1"))
            LOG.info("mlflow.dagshub.connected", uri=tracking_uri)  # log without creds
            return True
        except Exception as e:
            LOG.error("dagshub.connection_failed", error=str(e) or e.__class__.__name__)
            raise RuntimeError(
                f"DagsHub MLflow connection failed: {e}\n"
                "Verify your DAGSHUB_USER_TOKEN is correct and the repo exists at:\n"
                f"  {tracking_uri}"
            ) from e
    return _local_mlflow(mlflow, mlflow_cfg)


def _local_mlflow(mlflow, mlflow_cfg: dict) -> bool:
    local_uri = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "cloudscale-ppo-phase1-local"))
    LOG.info("mlflow.local", uri=local_uri)
    return True


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(args, cfg: dict) -> dict[str, float]:
    """Train PPO once with the (possibly Optuna-overridden) config."""
    import mlflow
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

    # MLflow logging
    if args.use_mlflow:
        run_name = f"{cfg.get('mlflow', {}).get('run_name_prefix', 'ppo')}-seed{args.seed}-{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in ppo_kwargs.items() if not isinstance(v, dict)})
            mlflow.log_params(asdict(env_cfg))
            mlflow.log_metrics(metrics)
            if cfg.get("mlflow", {}).get("log_model_artifact", True):
                model_path = MODELS_DIR / f"ppo_seed{args.seed}.zip"
                model.save(model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                LOG.info("model.saved", path=str(model_path))
    else:
        model_path = MODELS_DIR / f"ppo_seed{args.seed}.zip"
        model.save(model_path)
        LOG.info("model.saved_local", path=str(model_path))

    return metrics


# ---------------------------------------------------------------------------
# Optuna sweep
# ---------------------------------------------------------------------------

def train_optuna(args, cfg: dict) -> dict[str, Any]:
    import mlflow
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

        if args.use_mlflow:
            with mlflow.start_run(run_name=f"optuna-trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metrics(metrics)
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
    if args.use_mlflow:
        with mlflow.start_run(run_name="optuna-summary"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_value", study.best_value)
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
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    p.set_defaults(use_mlflow=True)
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
    setup_mlflow(cfg, use_mlflow=args.use_mlflow)

    if args.mode == "single":
        result = train_single(args, cfg)
    else:
        result = train_optuna(args, cfg)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
