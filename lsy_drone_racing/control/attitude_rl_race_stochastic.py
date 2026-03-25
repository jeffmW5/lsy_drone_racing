"""Stochastic RL controller — deploys with configurable exploration noise.

Same as attitude_rl_race.py but samples from the learned distribution
instead of using the deterministic mean. Noise scale controlled by
DRONE_RL_NOISE_SCALE env var (default 1.0 = full learned std).

Usage:
    DRONE_RL_CKPT_PATH=/path/to/model.ckpt DRONE_RL_NOISE_SCALE=0.3 \
        pixi run python scripts/sim_midair.py --config level2_attitude.toml \
        --controller attitude_rl_race_stochastic.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from torch.distributions import Normal

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """Stochastic RL controller for RaceCoreEnv-trained models."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

        self.n_obs = 2
        self.n_gates = len(config.env.track.gates)
        obs_dim = 13 + 3 + 4 + 3 + 4 + 13 * self.n_obs + 4  # = 57

        model_path = os.environ.get("DRONE_RL_CKPT_PATH")
        if not model_path:
            raise ValueError("DRONE_RL_CKPT_PATH environment variable not set.")

        self.noise_scale = float(os.environ.get("DRONE_RL_NOISE_SCALE", "1.0"))

        self.agent = Agent((obs_dim,), (4,)).to("cpu")
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        # Print noise info
        logstd = self.agent.actor_logstd.detach()
        std = torch.exp(logstd).squeeze()
        print(f"[Stochastic] noise_scale={self.noise_scale}, learned_std={std.numpy()}, "
              f"effective_std={std.numpy() * self.noise_scale}")

        self.last_action = np.zeros(4, dtype=np.float32)
        basic_obs = np.concatenate([obs[k] for k in ["pos", "quat", "vel", "ang_vel"]], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        obs_flat = self._preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_mean = self.agent.actor_mean(obs_tensor)
            action_logstd = self.agent.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd) * self.noise_scale

            if self.noise_scale > 0:
                probs = Normal(action_mean, action_std)
                raw_action = probs.sample().squeeze(0).numpy().copy()
            else:
                raw_action = action_mean.squeeze(0).numpy().copy()

        raw_action = np.clip(raw_action, -1.0, 1.0)
        self.last_action = raw_action.copy()
        raw_action[2] = 0.0
        return self._scale_action(raw_action).astype(np.float32)

    def _preprocess_obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        drone_pos = obs["pos"]
        target_gate_idx = int(obs["target_gate"])
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]

        safe_target = np.clip(target_gate_idx, 0, self.n_gates - 1)
        next_target = np.clip(target_gate_idx + 1, 0, self.n_gates - 1)

        rel_target = gates_pos[safe_target] - drone_pos
        target_quat = gates_quat[safe_target]
        rel_next = gates_pos[next_target] - drone_pos
        next_quat = gates_quat[next_target]

        basic_obs = np.concatenate(
            [obs[k] for k in ["pos", "quat", "vel", "ang_vel"]], axis=-1
        )

        parts = [
            obs["pos"], obs["quat"], obs["vel"], obs["ang_vel"],
            rel_target, target_quat, rel_next, next_quat,
            self.prev_obs.reshape(-1), self.last_action,
        ]
        flat = np.concatenate(parts, axis=-1).astype(np.float32)

        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], basic_obs[None, :]], axis=0
        )
        return flat

    def _scale_action(self, action: NDArray) -> NDArray:
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        center = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0) * scale + center

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        self.prev_obs = np.zeros_like(self.prev_obs)
        self.last_action = np.zeros(4, dtype=np.float32)
        self._finished = False
