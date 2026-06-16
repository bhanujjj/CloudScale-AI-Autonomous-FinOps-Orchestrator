"""
CloudScale K8s + Spot Gymnasium Environment (Multi-Zone Arbitrage)
==================================================================

A custom Gymnasium environment that simulates a Kubernetes cluster with
mixed Spot/On-Demand node pools across 3 Availability Zones. The agent must 
learn "Spot Arbitrage": shifting workloads between zones to avoid high prices
and reclaim events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class K8sSpotConfig:
    # Capacity per node
    cpu_per_node: float = 100.0
    mem_per_node: float = 100.0

    # Pricing (USD per simulated hour).
    ondemand_price_per_hr: float = 0.50
    spot_base_prices: list[float] = field(default_factory=lambda: [0.18, 0.16, 0.20])
    spot_price_volatilities: list[float] = field(default_factory=lambda: [0.20, 0.35, 0.15])

    # Cluster bounds
    min_nodes: int = 1
    max_nodes: int = 20

    # SLA
    sla_p95_latency_ms: float = 250.0
    sla_penalty_weight: float = 5.0

    # Reward shaping
    cost_weight: float = 1.0
    churn_penalty: float = 0.02
    sla_bonus: float = 0.01
    interrupt_penalty: float = 0.50
    cpu_safe_limit: float = 0.80
    cpu_penalty_weight: float = 2.0

    # Workload dynamics (Chaotic)
    request_rate_base: float = 300.0
    request_rate_amplitude: float = 250.0
    request_rate_noise_std: float = 20.0
    flash_sale_probability: float = 0.005
    flash_sale_duration_mean: float = 45.0
    flash_sale_multiplier: float = 4.0

    # Latency model
    latency_baseline_ms: float = 40.0
    latency_pressure_coeff: float = 1500.0

    # Spot interruption model
    spot_interrupt_base_probs: list[float] = field(default_factory=lambda: [0.003, 0.008, 0.001])
    spot_interrupt_price_threshold: float = 1.8

    # Episode
    episode_length_steps: int = 500
    step_duration_minutes: float = 1.0

    # Reproducibility
    seed: int | None = None


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------

class K8sSpotEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    OBS_DIM = 14
    NUM_ZONES = 3

    # Action constants
    NO_OP = 0
    ADD_ONDEMAND = 1
    REMOVE_ONDEMAND = 2
    ADD_SPOT_A = 3
    REMOVE_SPOT_A = 4
    ADD_SPOT_B = 5
    REMOVE_SPOT_B = 6
    ADD_SPOT_C = 7
    REMOVE_SPOT_C = 8

    def __init__(self, config: K8sSpotConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or K8sSpotConfig()
        self.render_mode = render_mode

        # Action & observation spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

        # State
        self.rng: np.random.Generator = np.random.default_rng(self.cfg.seed)
        self.t: int = 0
        self.spot_nodes: list[int] = [0] * self.NUM_ZONES
        self.ondemand_nodes: int = 0
        self.spot_prices: list[float] = list(self.cfg.spot_base_prices)
        
        self.last_request_rate: float = 0.0
        self.last_cpu_util: float = 0.0
        self.last_mem_util: float = 0.0
        self.last_p95_ms: float = 0.0
        self.sla_violation_window: list[bool] = []
        self.last_action_was_noop: bool = True
        
        self.episode_cost: float = 0.0
        self.episode_sla_violations: int = 0
        self.episode_headroom_breaches: int = 0
        self.flash_sale_steps_left: int = 0

    def _current_request_rate(self) -> float:
        hours = (self.t * self.cfg.step_duration_minutes) / 60.0
        daily = self.cfg.request_rate_base + self.cfg.request_rate_amplitude * math.sin(
            2 * math.pi * (hours - 6.0) / 24.0
        )
        
        if self.flash_sale_steps_left > 0:
            self.flash_sale_steps_left -= 1
            multiplier = self.cfg.flash_sale_multiplier
        else:
            multiplier = 1.0
            if self.rng.random() < self.cfg.flash_sale_probability:
                duration = int(self.rng.normal(self.cfg.flash_sale_duration_mean, 10.0))
                self.flash_sale_steps_left = max(5, duration)
                multiplier = self.cfg.flash_sale_multiplier

        noise = self.rng.normal(0.0, self.cfg.request_rate_noise_std)
        rate = max(0.0, daily + noise) * multiplier
        return rate

    def _total_capacity(self) -> float:
        return (sum(self.spot_nodes) + self.ondemand_nodes) * self.cfg.cpu_per_node

    def _utilization(self, demand: float) -> tuple[float, float, float]:
        cap = max(self._total_capacity(), 1e-3)
        cpu_load = min(demand, cap)
        mem_load = min(demand * 0.85, cap)
        cpu_util = float(np.clip(cpu_load / cap + self.rng.normal(0, 0.02), 0, 1.5))
        mem_util = float(np.clip(mem_load / cap + self.rng.normal(0, 0.02), 0, 1.5))

        pressure = max(0.0, cpu_util - 0.7)
        latency = self.cfg.latency_baseline_ms + self.cfg.latency_pressure_coeff * (pressure ** 2)
        latency += self.rng.normal(0, 5.0)
        latency = max(1.0, latency)
        return cpu_util, mem_util, latency

    def _update_spot_prices(self) -> None:
        for i in range(self.NUM_ZONES):
            base = self.cfg.spot_base_prices[i]
            drift = (base - self.spot_prices[i]) * 0.05
            shock = self.rng.normal(0.0, base * self.cfg.spot_price_volatilities[i] * 0.1)
            self.spot_prices[i] = max(base * 0.4, min(base * 3.0, self.spot_prices[i] + drift + shock))

    def _maybe_interrupt_spot(self) -> int:
        interrupts = 0
        for i in range(self.NUM_ZONES):
            if self.spot_nodes[i] <= 0:
                continue
            price_ratio = self.spot_prices[i] / self.cfg.spot_base_prices[i]
            prob = self.cfg.spot_interrupt_base_probs[i]
            if price_ratio > self.cfg.spot_interrupt_price_threshold:
                prob *= 4.0
            if self.rng.random() < prob:
                self.spot_nodes[i] -= 1
                interrupts += 1
        return interrupts

    def _apply_action(self, action: int) -> tuple[bool, str]:
        total = sum(self.spot_nodes) + self.ondemand_nodes
        reason = ""

        if action == self.NO_OP:
            return False, ""
        elif action == self.ADD_ONDEMAND:
            if total >= self.cfg.max_nodes: return False, "max_nodes_reached"
            self.ondemand_nodes += 1
        elif action == self.REMOVE_ONDEMAND:
            if self.ondemand_nodes <= 0: return False, "no_ondemand_to_remove"
            self.ondemand_nodes -= 1
        elif action == self.ADD_SPOT_A:
            if total >= self.cfg.max_nodes: return False, "max_nodes_reached"
            self.spot_nodes[0] += 1
        elif action == self.REMOVE_SPOT_A:
            if self.spot_nodes[0] <= 0: return False, "no_spot_to_remove"
            self.spot_nodes[0] -= 1
        elif action == self.ADD_SPOT_B:
            if total >= self.cfg.max_nodes: return False, "max_nodes_reached"
            self.spot_nodes[1] += 1
        elif action == self.REMOVE_SPOT_B:
            if self.spot_nodes[1] <= 0: return False, "no_spot_to_remove"
            self.spot_nodes[1] -= 1
        elif action == self.ADD_SPOT_C:
            if total >= self.cfg.max_nodes: return False, "max_nodes_reached"
            self.spot_nodes[2] += 1
        elif action == self.REMOVE_SPOT_C:
            if self.spot_nodes[2] <= 0: return False, "no_spot_to_remove"
            self.spot_nodes[2] -= 1
        else:
            return False, f"unknown_action_{action}"

        return True, ""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.cfg.seed = seed
        self.rng = np.random.default_rng(self.cfg.seed)

        self.t = 0
        self.spot_nodes = [1, 1, 0]
        self.ondemand_nodes = 2
        self.spot_prices = list(self.cfg.spot_base_prices)
        self.last_request_rate = self._current_request_rate()
        cpu, mem, lat = self._utilization(self.last_request_rate)
        self.last_cpu_util, self.last_mem_util, self.last_p95_ms = cpu, mem, lat
        self.sla_violation_window = []
        self.last_action_was_noop = True
        self.episode_cost = 0.0
        self.episode_sla_violations = 0
        self.episode_headroom_breaches = 0
        self.flash_sale_steps_left = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        was_churn, invalid_reason = self._apply_action(int(action))

        self._update_spot_prices()
        interrupts = self._maybe_interrupt_spot()

        self.t += 1
        demand = self._current_request_rate()
        cpu, mem, lat = self._utilization(demand)
        self.last_request_rate = demand
        self.last_cpu_util, self.last_mem_util, self.last_p95_ms = cpu, mem, lat

        step_hours = self.cfg.step_duration_minutes / 60.0
        spot_cost = sum(n * p for n, p in zip(self.spot_nodes, self.spot_prices))
        cost = (spot_cost + self.ondemand_nodes * self.cfg.ondemand_price_per_hr) * step_hours * self.cfg.cost_weight
        self.episode_cost += cost

        sla_breach = lat > self.cfg.sla_p95_latency_ms
        self.sla_violation_window.append(sla_breach)
        if len(self.sla_violation_window) > 50:
            self.sla_violation_window.pop(0)
        if sla_breach:
            self.episode_sla_violations += 1

        reward = -cost
        if sla_breach:
            overage = (lat - self.cfg.sla_p95_latency_ms) / self.cfg.sla_p95_latency_ms
            reward -= self.cfg.sla_penalty_weight * (overage ** 2)
        else:
            reward += self.cfg.sla_bonus
            
        if cpu > self.cfg.cpu_safe_limit:
            self.episode_headroom_breaches += 1
            reward -= self.cfg.cpu_penalty_weight * (cpu - self.cfg.cpu_safe_limit)

        if was_churn:
            reward -= self.cfg.churn_penalty
        if interrupts > 0:
            reward -= self.cfg.interrupt_penalty * interrupts

        terminated = self.t >= self.cfg.episode_length_steps
        info = self._get_info()
        info.update({"cost": cost, "interrupted": interrupts > 0, "invalid_action": invalid_reason, "sla_breach": sla_breach})

        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        hours = (self.t * self.cfg.step_duration_minutes) / 60.0
        h_frac = (hours % 24.0) / 24.0
        sla_viol_rate = float(np.mean(self.sla_violation_window)) if self.sla_violation_window else 0.0
        cap = max(self._total_capacity(), 1e-3)
        
        obs = [
            float(np.clip(self.last_cpu_util, 0, 1)),
            float(np.clip(self.last_mem_util, 0, 1)),
            float(np.clip(self.last_request_rate / cap, 0, 1)),
            float(np.clip(self.last_p95_ms / (self.cfg.sla_p95_latency_ms * 2), 0, 1)),
            self.ondemand_nodes / self.cfg.max_nodes,
            self.spot_nodes[0] / self.cfg.max_nodes,
            self.spot_nodes[1] / self.cfg.max_nodes,
            self.spot_nodes[2] / self.cfg.max_nodes,
            float(np.clip(self.spot_prices[0] / (self.cfg.spot_base_prices[0] * 2), 0, 1)),
            float(np.clip(self.spot_prices[1] / (self.cfg.spot_base_prices[1] * 2), 0, 1)),
            float(np.clip(self.spot_prices[2] / (self.cfg.spot_base_prices[2] * 2), 0, 1)),
            sla_viol_rate,
            math.sin(2 * math.pi * h_frac),
            math.cos(2 * math.pi * h_frac),
        ]
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "spot_nodes_A": self.spot_nodes[0],
            "spot_nodes_B": self.spot_nodes[1],
            "spot_nodes_C": self.spot_nodes[2],
            "ondemand_nodes": self.ondemand_nodes,
            "spot_price_A": self.spot_prices[0],
            "spot_price_B": self.spot_prices[1],
            "spot_price_C": self.spot_prices[2],
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
                f"t={self.t:4d} | spot=[{self.spot_nodes[0]},{self.spot_nodes[1]},{self.spot_nodes[2]}] od={self.ondemand_nodes:2d} "
                f"px=[${self.spot_prices[0]:.2f},${self.spot_prices[1]:.2f},${self.spot_prices[2]:.2f}] | cpu={self.last_cpu_util:.2f} "
                f"p95={self.last_p95_ms:6.1f}ms cost=${self.episode_cost:.2f} sla={self.episode_sla_violations}"
            )
            return line
        return None

    def close(self) -> None:
        pass


def make_env(config: K8sSpotConfig | None = None) -> K8sSpotEnv:
    return K8sSpotEnv(config=config)

if __name__ == "__main__":
    env = K8sSpotEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("Initial obs:", obs)
    for i in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"step {i:2d}  action={a}  reward={r:+.4f}  " + env.render())
        if term or trunc: break
