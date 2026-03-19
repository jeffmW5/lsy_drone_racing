"""RL controller for models trained on VecDroneRaceEnv (RaceCoreEnv pipeline).

Reads configuration from environment variables:
    DRONE_RL_CKPT_PATH  - path to model.ckpt (required)

Unlike attitude_rl_generic.py, this controller does NOT build a trajectory.
Instead, it preprocesses the obs dict the same way as training:
    drone state + relative gate positions → flat vector → agent forward pass

Usage:
    DRONE_RL_CKPT_PATH=/path/to/model.ckpt pixi run python scripts/sim.py \
        --config level2_attitude.toml --controller attitude_rl_race.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """RL controller for RaceCoreEnv-trained models."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

        self.n_obs = 2
        self.n_gates = len(config.env.track.gates)

        # Obs dims: pos(3) + quat(4) + vel(3) + ang_vel(3) + rel_target(3) +
        #           target_quat(4) + rel_next(3) + next_quat(4) + prev_obs(26) + last_action(4)
        obs_dim = 13 + 3 + 4 + 3 + 4 + 13 * self.n_obs + 4  # = 57

        # Load RL policy
        model_path = os.environ.get("DRONE_RL_CKPT_PATH")
        if not model_path:
            raise ValueError(
                "DRONE_RL_CKPT_PATH environment variable not set. "
                "Set it to the path of the model.ckpt file."
            )
        self.agent = Agent((obs_dim,), (4,)).to("cpu")
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        # State tracking
        self.last_action = np.zeros(4, dtype=np.float32)
        basic_obs = np.concatenate([obs[k] for k in ["pos", "quat", "vel", "ang_vel"]], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))  # (n_obs, 13)

        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        obs_flat = self._preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            act, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)
            raw_action = act.squeeze(0).numpy().copy()

        # Store raw action for obs preprocessing (before yaw zeroing)
        self.last_action = raw_action.copy()

        # Zero out yaw (matching training NormalizeRaceActions)
        raw_action[2] = 0.0

        # Scale from [-1, 1] to actual attitude space
        return self._scale_action(raw_action).astype(np.float32)

    def _preprocess_obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Preprocess obs to match training pipeline exactly.

        Must match: RaceRewardAndObs → RaceStackObs → ActionPenalty → Flatten
        Output order: pos, quat, vel, ang_vel, rel_target_gate, target_gate_quat,
                      rel_next_gate, next_gate_quat, prev_obs, last_action
        """
        drone_pos = obs["pos"]  # (3,)
        target_gate_idx = int(obs["target_gate"])
        gates_pos = obs["gates_pos"]  # (n_gates, 3)
        gates_quat = obs["gates_quat"]  # (n_gates, 4)

        # Clamp to valid range
        safe_target = np.clip(target_gate_idx, 0, self.n_gates - 1)
        next_target = np.clip(target_gate_idx + 1, 0, self.n_gates - 1)

        # Relative gate positions
        rel_target = gates_pos[safe_target] - drone_pos
        target_quat = gates_quat[safe_target]
        rel_next = gates_pos[next_target] - drone_pos
        next_quat = gates_quat[next_target]

        # Update obs history
        basic_obs = np.concatenate(
            [obs[k] for k in ["pos", "quat", "vel", "ang_vel"]], axis=-1
        )  # (13,)

        # Build flat observation (same order as FlattenJaxObservation)
        parts = [
            obs["pos"],           # 3
            obs["quat"],          # 4
            obs["vel"],           # 3
            obs["ang_vel"],       # 3
            rel_target,           # 3
            target_quat,          # 4
            rel_next,             # 3
            next_quat,            # 4
            self.prev_obs.reshape(-1),  # 13 * n_obs
            self.last_action,     # 4
        ]
        flat = np.concatenate(parts, axis=-1).astype(np.float32)

        # Update history buffer (shift and append current)
        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], basic_obs[None, :]], axis=0
        )

        return flat

    def _scale_action(self, action: NDArray) -> NDArray:
        """Scale normalized [-1, 1] action to actual attitude command."""
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        center = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0) * scale + center

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return self._finished

    def episode_callback(self):
        self.prev_obs = np.zeros_like(self.prev_obs)
        self.last_action = np.zeros(4, dtype=np.float32)
        self._finished = False
