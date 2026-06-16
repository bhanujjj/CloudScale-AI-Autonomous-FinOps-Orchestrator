"""
CloudScale K8s + Spot Gymnasium Environment
============================================

A custom Gymnasium environment that simulates a Kubernetes cluster with
mixed Spot/On-Demand node pools. The agent must learn to scale the cluster
to handle variable request load while minimizing cost and respecting an
SLA on p95 latency.

Why this exists
---------------
This is the core of Phase 1: a faithful simulator that lets us train a
Reinforcement Learning (PPO) policy on a MacBook CPU in minutes, before
we ever touch a real cloud cluster. The simulator captures the *tricky*
trade-offs that make this problem interesting:

  1. Spot instances are 60-70% cheaper but can be reclaimed with ~2-minute
     notice. Aggressive cost optimization means risking SLA violations.
  2. On-Demand is expensive but rock-solid. A smart agent only uses it
     when (a) load is high, or (b) spot risk is elevated.
  3. Node churn has a real cost: scaling up/down too often wastes both
     money and time. The agent must learn patience.

Observation space (Box, 10 dims, all normalized to roughly [0, 1])
-----------------------------------------------------------------
  0  cpu_utilization        rolling mean CPU across all pods
  1  memory_utilization     rolling mean memory across all pods
  2  request_rate           requests/sec, normalized to max capacity
  3  p95_latency_ms         normalized against the SLA threshold
  4  spot_node_count        normalized to max_nodes
  5  ondemand_node_count    normalized to max_nodes
  6  spot_price_multiplier  current spot price / base spot price
  7  sla_violation_rate     fraction of recent steps where SLA was breached
  8  hour_of_day_sin        cyclical encoding (sin)
  9  hour_of_day_cos        cyclical encoding (cos)

Action space (Discrete, 6 actions)
----------------------------------
  0  no_op
  1  add_spot               - try to be aggressive on cost
  2  add_ondemand           - safe scale up under load
  3  remove_spot            - cut cost when idle
  4  spot_to_ondemand       - defensive shift under high risk
  5  ondemand_to_spot       - aggressive cost rebalance when safe

Reward function (per step, summed over episode)
-----------------------------------------------
  cost           = -$  (linear in node count * price)
  sla_penalty    = -$  (huge if p95 latency breaches SLA)
  churn_penalty  = -$  (small, discourages thrashing)
  sla_bonus      = +$  (small, reinforces SLA compliance)
  interrupt_hit  = -$  (large, when a spot reclaim happens)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Configuration dataclass — easier to swap into Optuna later than a dict
# ---------------------------------------------------------------------------

@dataclass
class K8sSpotConfig:
    # Capacity per node (in arbitrary load units)
    cpu_per_node: float = 100.0          # each node can handle 100 CPU units
    mem_per_node: float = 100.0          # each node can hold 100 mem units

    # Pricing (USD per simulated hour). Tuned to roughly match Civo/GCP.
    ondemand_price_per_hr: float = 0.50
    spot_base_price_per_hr: float = 0.18  # ~64% discount
    spot_price_volatility: float = 0.20   # 20% std dev random walk

    # Cluster bounds
    min_nodes: int = 1
    max_nodes: int = 20

    # SLA
    sla_p95_latency_ms: float = 250.0
    sla_penalty_weight: float = 5.0       # multiplier on overage

    # Reward shaping
    cost_weight: float = 1.0
    churn_penalty: float = 0.02
    sla_bonus: float = 0.01
    interrupt_penalty: float = 0.50
    cpu_safe_limit: float = 0.80
    cpu_penalty_weight: float = 2.0

    # Workload dynamics
    request_rate_base: float = 50.0       # requests/sec at midday baseline
    request_rate_amplitude: float = 30.0  # +/- around base over a day
    request_rate_noise_std: float = 5.0
    burst_probability: float = 0.02       # chance per step of a 3x spike
    burst_multiplier: float = 3.0

    # Latency model
    latency_baseline_ms: float = 40.0
    latency_pressure_coeff: float = 1500.0  # how fast latency explodes with overload

    # Spot interruption model
    spot_interrupt_base_prob: float = 0.003   # per step (≈10% per hour)
    spot_interrupt_price_threshold: float = 1.8  # above this, risk spikes

    # Episode
    episode_length_steps: int = 500
    step_duration_minutes: float = 1.0     # each env step == 1 sim minute

    # Reproducibility
    seed: int | None = None


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------

class K8sSpotEnv(gym.Env):
    """A K8s cluster with Spot/On-Demand nodes. See module docstring."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    OBS_DIM = 10

    # Action constants — handy for callers / tests
    NO_OP = 0
    ADD_SPOT = 1
    ADD_ONDEMAND = 2
    REMOVE_SPOT = 3
    SPOT_TO_ONDEMAND = 4
    ONDEMAND_TO_SPOT = 5

    def __init__(self, config: K8sSpotConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or K8sSpotConfig()
        self.render_mode = render_mode

        # Action & observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

        # State — populated in reset()
        self.rng: np.random.Generator = np.random.default_rng(self.cfg.seed)
        self.t: int = 0
        self.spot_nodes: int = 0
        self.ondemand_nodes: int = 0
        self.spot_price: float = self.cfg.spot_base_price_per_hr
        self.last_request_rate: float = 0.0
        self.last_cpu_util: float = 0.0
        self.last_mem_util: float = 0.0
        self.last_p95_ms: float = 0.0
        self.sla_violation_window: list[bool] = []  # rolling window for violation rate
        self.last_action_was_noop: bool = True
        self.episode_cost: float = 0.0
        self.episode_sla_violations: int = 0
        self.episode_headroom_breaches: int = 0

    # ------------------------------------------------------------------ helpers
    def _current_request_rate(self) -> float:
        """Daily sinusoid + Gaussian noise + occasional bursts."""
        # hour_of_day in [0, 24), cycles over episode
        hours = (self.t * self.cfg.step_duration_minutes) / 60.0
        daily = self.cfg.request_rate_base + self.cfg.request_rate_amplitude * math.sin(
            2 * math.pi * (hours - 6.0) / 24.0
        )
        noise = self.rng.normal(0.0, self.cfg.request_rate_noise_std)
        rate = max(0.0, daily + noise)
        if self.rng.random() < self.cfg.burst_probability:
            rate *= self.cfg.burst_multiplier
        return rate

    def _total_capacity(self) -> float:
        return (self.spot_nodes + self.ondemand_nodes) * self.cfg.cpu_per_node

    def _utilization(self, demand: float) -> tuple[float, float, float]:
        """Returns (cpu_util, mem_util, p95_latency_ms) given a demand level."""
        cap = max(self._total_capacity(), 1e-3)
        # Request rate acts as "load units" per second. CPU & mem track it
        # with a small independent noise.
        cpu_load = min(demand, cap)
        mem_load = min(demand * 0.85, cap)  # mem typically a bit lower than cpu
        cpu_util = float(np.clip(cpu_load / cap + self.rng.normal(0, 0.02), 0, 1.5))
        mem_util = float(np.clip(mem_load / cap + self.rng.normal(0, 0.02), 0, 1.5))

        # Latency explodes non-linearly as utilization passes ~0.7
        pressure = max(0.0, cpu_util - 0.7)
        latency = self.cfg.latency_baseline_ms + self.cfg.latency_pressure_coeff * (pressure ** 2)
        latency += self.rng.normal(0, 5.0)
        latency = max(1.0, latency)
        return cpu_util, mem_util, latency

    def _update_spot_price(self) -> None:
        """Random-walk spot price, mean-reverting to the base."""
        drift = (self.cfg.spot_base_price_per_hr - self.spot_price) * 0.05
        shock = self.rng.normal(0.0, self.cfg.spot_base_price_per_hr * self.cfg.spot_price_volatility * 0.1)
        self.spot_price = max(
            self.cfg.spot_base_price_per_hr * 0.4,
            min(self.cfg.spot_base_price_per_hr * 3.0, self.spot_price + drift + shock),
        )

    def _maybe_interrupt_spot(self) -> bool:
        """Returns True if a spot node was reclaimed this step."""
        if self.spot_nodes <= 0:
            return False
        price_ratio = self.spot_price / self.cfg.spot_base_price_per_hr
        # Higher price → higher interruption probability (market is "hot")
        prob = self.cfg.spot_interrupt_base_prob
        if price_ratio > self.cfg.spot_interrupt_price_threshold:
            prob *= 4.0
        if self.rng.random() < prob:
            # Reclaim 1 spot node
            self.spot_nodes -= 1
            return True
        return False

    def _apply_action(self, action: int) -> tuple[bool, str]:
        """Mutate node counts. Returns (was_churn, reason_if_invalid)."""
        total = self.spot_nodes + self.ondemand_nodes
        churn = True  # default: we count any non-noop as churn
        reason = ""

        if action == self.NO_OP:
            return False, ""

        if action == self.ADD_SPOT:
            if total >= self.cfg.max_nodes:
                reason = "max_nodes_reached"
            else:
                self.spot_nodes += 1
                return True, ""

        elif action == self.ADD_ONDEMAND:
            if total >= self.cfg.max_nodes:
                reason = "max_nodes_reached"
            else:
                self.ondemand_nodes += 1
                return True, ""

        elif action == self.REMOVE_SPOT:
            if self.spot_nodes <= 0:
                reason = "no_spot_to_remove"
            else:
                self.spot_nodes -= 1
                return True, ""

        elif action == self.SPOT_TO_ONDEMAND:
            if self.spot_nodes <= 0:
                reason = "no_spot_to_shift"
            else:
                self.spot_nodes -= 1
                self.ondemand_nodes += 1
                return True, ""

        elif action == self.ONDEMAND_TO_SPOT:
            if self.ondemand_nodes <= 0:
                reason = "no_ondemand_to_shift"
            else:
                self.ondemand_nodes -= 1
                self.spot_nodes += 1
                return True, ""
        else:
            reason = f"unknown_action_{action}"

        # We hit an invalid path; count it as non-churn (no state change)
        return False, reason

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.cfg.seed = seed
        self.rng = np.random.default_rng(self.cfg.seed)

        self.t = 0
        # Start with a small mixed cluster
        self.spot_nodes = 2
        self.ondemand_nodes = 1
        self.spot_price = self.cfg.spot_base_price_per_hr
        self.last_request_rate = self._current_request_rate()
        cpu, mem, lat = self._utilization(self.last_request_rate)
        self.last_cpu_util, self.last_mem_util, self.last_p95_ms = cpu, mem, lat
        self.sla_violation_window = []
        self.last_action_was_noop = True
        self.episode_cost = 0.0
        self.episode_sla_violations = 0
        self.episode_headroom_breaches = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1. Apply action
        was_churn, invalid_reason = self._apply_action(int(action))

        # 2. Apply spot dynamics: price walk, possible interruption
        self._update_spot_price()
        interrupted = self._maybe_interrupt_spot()

        # 3. Advance workload & recompute metrics
        self.t += 1
        demand = self._current_request_rate()
        cpu, mem, lat = self._utilization(demand)
        self.last_request_rate = demand
        self.last_cpu_util, self.last_mem_util, self.last_p95_ms = cpu, mem, lat

        # 4. Compute cost (per-hour price * step fraction)
        step_hours = self.cfg.step_duration_minutes / 60.0
        cost = (
            self.spot_nodes * self.spot_price
            + self.ondemand_nodes * self.cfg.ondemand_price_per_hr
        ) * step_hours * self.cfg.cost_weight
        self.episode_cost += cost

        # 5. SLA tracking
        sla_breach = lat > self.cfg.sla_p95_latency_ms
        self.sla_violation_window.append(sla_breach)
        if len(self.sla_violation_window) > 50:
            self.sla_violation_window.pop(0)
        if sla_breach:
            self.episode_sla_violations += 1

        # 6. Reward
        reward = -cost
        if sla_breach:
            # Quadratic overage — punishing severe breaches
            overage = (lat - self.cfg.sla_p95_latency_ms) / self.cfg.sla_p95_latency_ms
            reward -= self.cfg.sla_penalty_weight * (overage ** 2)
        else:
            reward += self.cfg.sla_bonus
            
        headroom_breach = cpu > self.cfg.cpu_safe_limit
        if headroom_breach:
            self.episode_headroom_breaches += 1
            cpu_overage = cpu - self.cfg.cpu_safe_limit
            reward -= self.cfg.cpu_penalty_weight * cpu_overage

        if was_churn:
            reward -= self.cfg.churn_penalty
        if interrupted:
            reward -= self.cfg.interrupt_penalty

        # 7. Termination
        terminated = self.t >= self.cfg.episode_length_steps
        truncated = False
        if invalid_reason:
            truncated = False  # not truncation, just info; invalid actions are silently no-op'd

        info = self._get_info()
        info["cost"] = cost
        info["interrupted"] = interrupted
        info["invalid_action"] = invalid_reason
        info["sla_breach"] = sla_breach
        info["latency_ms"] = lat

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        hours = (self.t * self.cfg.step_duration_minutes) / 60.0
        h_frac = (hours % 24.0) / 24.0
        sla_viol_rate = (
            float(np.mean(self.sla_violation_window)) if self.sla_violation_window else 0.0
        )
        cap = max(self._total_capacity(), 1e-3)
        obs = np.array(
            [
                float(np.clip(self.last_cpu_util, 0, 1)),
                float(np.clip(self.last_mem_util, 0, 1)),
                float(np.clip(self.last_request_rate / cap, 0, 1)),
                float(np.clip(self.last_p95_ms / (self.cfg.sla_p95_latency_ms * 2), 0, 1)),
                self.spot_nodes / self.cfg.max_nodes,
                self.ondemand_nodes / self.cfg.max_nodes,
                float(np.clip(self.spot_price / (self.cfg.spot_base_price_per_hr * 2), 0, 1)),
                sla_viol_rate,
                math.sin(2 * math.pi * h_frac),
                math.cos(2 * math.pi * h_frac),
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "spot_nodes": self.spot_nodes,
            "ondemand_nodes": self.ondemand_nodes,
            "spot_price": self.spot_price,
            "cpu_util": self.last_cpu_util,
            "mem_util": self.last_mem_util,
            "p95_latency_ms": self.last_p95_ms,
            "episode_cost": self.episode_cost,
            "episode_sla_violations": self.episode_sla_violations,
            "episode_headroom_breaches": self.episode_headroom_breaches,
        }

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            line = (
                f"t={self.t:4d} | spot={self.spot_nodes:2d} od={self.ondemand_nodes:2d} "
                f"px=${self.spot_price:.3f} | cpu={self.last_cpu_util:.2f} "
                f"mem={self.last_mem_util:.2f} p95={self.last_p95_ms:6.1f}ms "
                f"cost=${self.episode_cost:.2f} sla_breaches={self.episode_sla_violations}"
            )
            return line
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Convenience factory & self-test
# ---------------------------------------------------------------------------

def make_env(config: K8sSpotConfig | None = None) -> K8sSpotEnv:
    return K8sSpotEnv(config=config)


if __name__ == "__main__":
    # Smoke test: roll out 50 random steps and print state.
    env = K8sSpotEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("Initial obs:", obs)
    print("Initial info:", info)
    for i in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"step {i:2d}  action={a}  reward={r:+.4f}  " + env.render())
        if term or trunc:
            break
    print("\nFinal info:", info)
