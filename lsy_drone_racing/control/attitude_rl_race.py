"""RL controller for models trained on VecDroneRaceEnv (RaceCoreEnv pipeline).

Reads configuration from environment variables:
    DRONE_RL_CKPT_PATH  - path to model.ckpt (required)
    DRONE_RL_TAKEOFF_ALT - altitude threshold to switch from takeoff to RL (default: 0.4)
    DRONE_RL_TAKEOFF_STEPS - max steps for takeoff phase (default: 50)

Unlike attitude_rl_generic.py, this controller does NOT build a trajectory.
Instead, it preprocesses the obs dict the same way as training:
    drone state + relative gate positions → flat vector → agent forward pass

Includes a takeoff phase: if the drone starts near ground level, the controller
applies level hover thrust until reaching DRONE_RL_TAKEOFF_ALT, then switches
to the RL policy. This bridges the domain gap between mid-air training spawns
and ground-level benchmark starts.

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
        self.body_frame_obs = os.environ.get("DRONE_RL_BODY_FRAME_OBS", "").lower() == "true"

        if self.body_frame_obs:
            # Body-frame obs: pos(3)+quat(4)+vel(3)+ang_vel(3) + rel_body(3)+normal_body(3)×2
            #                 + prev_obs(26) + last_action(4) = 55
            obs_dim = 13 + 3 + 3 + 3 + 3 + 13 * self.n_obs + 4  # = 55
        else:
            # World-frame obs: pos(3)+quat(4)+vel(3)+ang_vel(3) + rel(3)+quat(4)×2
            #                  + prev_obs(26) + last_action(4) = 57
            obs_dim = 13 + 3 + 4 + 3 + 4 + 13 * self.n_obs + 4  # = 57

        # Load RL policy
        model_path = os.environ.get("DRONE_RL_CKPT_PATH")
        if not model_path:
            raise ValueError(
                "DRONE_RL_CKPT_PATH environment variable not set. "
                "Set it to the path of the model.ckpt file."
            )

        # Load checkpoint — detect architecture from weights
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        # Infer hidden_size and obs_dim from checkpoint weights
        hidden_size = int(ckpt["critic.0.weight"].shape[0]) if "critic.0.weight" in ckpt else 64
        ckpt_obs_dim = int(ckpt["actor_mean.0.weight"].shape[1]) if "actor_mean.0.weight" in ckpt else obs_dim
        if ckpt_obs_dim != obs_dim:
            # Auto-detect body_frame_obs from checkpoint obs dimension
            self.body_frame_obs = (ckpt_obs_dim == 55)
            obs_dim = ckpt_obs_dim
        if "_actor_obs_dim" in ckpt:
            # Asymmetric agent: load only actor weights into standard Agent
            self.agent = Agent((obs_dim,), (4,), hidden_size=hidden_size).to("cpu")
            actor_state = {k: v for k, v in ckpt.items()
                          if k.startswith("actor") and not k.startswith("_")}
            self.agent.load_state_dict(actor_state, strict=False)
        else:
            self.agent = Agent((obs_dim,), (4,), hidden_size=hidden_size).to("cpu")
            self.agent.load_state_dict(ckpt)
        self.agent.eval()

        # State tracking
        self.last_action = np.zeros(4, dtype=np.float32)
        basic_obs = np.concatenate([obs[k] for k in ["pos", "quat", "vel", "ang_vel"]], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))  # (n_obs, 13)

        self._finished = False
        self._step = 0

        # Takeoff phase: if starting near ground, apply hover thrust until reaching altitude
        self._takeoff_alt = float(os.environ.get("DRONE_RL_TAKEOFF_ALT", "0.4"))
        self._takeoff_max_steps = int(os.environ.get("DRONE_RL_TAKEOFF_STEPS", "50"))
        # Compute hover thrust action: thrust = mass * g, then map to [-1, 1] action space
        hover_thrust = self.drone_mass * 9.81
        # Slightly above hover to climb
        climb_thrust = hover_thrust * 1.5
        # Map to [-1, 1]: action = (thrust - center) / scale
        thrust_scale = (self.thrust_max - self.thrust_min) / 2.0
        thrust_center = (self.thrust_max + self.thrust_min) / 2.0
        self._takeoff_thrust_action = np.clip(
            (climb_thrust - thrust_center) / thrust_scale, -1.0, 1.0
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        self._step += 1

        # Takeoff phase: level hover with climb thrust until reaching altitude
        drone_z = obs["pos"][2]
        if self._step <= self._takeoff_max_steps and drone_z < self._takeoff_alt:
            # Zero roll/pitch/yaw, climb thrust
            raw_action = np.array([0.0, 0.0, 0.0, self._takeoff_thrust_action],
                                  dtype=np.float32)
            self.last_action = raw_action.copy()
            # Still update obs history so RL policy gets smooth transition
            self._preprocess_obs(obs)
            raw_action[2] = 0.0
            return self._scale_action(raw_action).astype(np.float32)

        # RL policy phase
        obs_flat = self._preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)

        stochastic = os.environ.get("DRONE_RL_STOCHASTIC", "").lower() == "true"
        noise_scale = float(os.environ.get("DRONE_RL_NOISE_SCALE", "0"))
        with torch.no_grad():
            if noise_scale > 0:
                # Temperature-scaled sampling: mean + noise_scale * std * N(0,1)
                action_mean = self.agent.actor_mean(obs_tensor)
                action_logstd = self.agent.actor_logstd.expand_as(action_mean)
                if hasattr(self.agent, "max_logstd"):
                    action_logstd = torch.clamp(action_logstd, max=self.agent.max_logstd)
                action_std = torch.exp(action_logstd)
                noise = torch.randn_like(action_mean)
                act = action_mean + noise_scale * action_std * noise
            else:
                act, _, _, _ = self.agent.get_action_and_value(
                    obs_tensor, deterministic=not stochastic
                )
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
        if self.body_frame_obs:
            # Transform gate info to drone body frame
            drone_rot_inv = Rotation.from_quat(obs["quat"]).inv()
            rel_target_body = drone_rot_inv.apply(rel_target)
            rel_next_body = drone_rot_inv.apply(rel_next)
            # Gate normals: x-axis of gate rotation, in body frame
            target_fwd = Rotation.from_quat(target_quat).apply([1.0, 0.0, 0.0])
            next_fwd = Rotation.from_quat(next_quat).apply([1.0, 0.0, 0.0])
            target_normal_body = drone_rot_inv.apply(target_fwd)
            next_normal_body = drone_rot_inv.apply(next_fwd)
            parts = [
                obs["pos"],               # 3
                obs["quat"],              # 4
                obs["vel"],               # 3
                obs["ang_vel"],           # 3
                rel_target_body,          # 3
                target_normal_body,       # 3
                rel_next_body,            # 3
                next_normal_body,         # 3
                self.prev_obs.reshape(-1),  # 13 * n_obs
                self.last_action,         # 4
            ]
        else:
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
        # Log trajectory for debugging
        if os.environ.get("DRONE_RL_LOG_TRAJECTORY"):
            pos = obs["pos"]
            vel = obs["vel"]
            gate_idx = int(obs.get("target_gate", 0))
            if self._step % 5 == 0 or terminated:
                status = "CRASH" if terminated else "fly"
                print(f"  step={self._step:4d} pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}] "
                      f"vel=[{vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}] "
                      f"gate={gate_idx} {status}")
        return self._finished

    def episode_callback(self):
        self.prev_obs = np.zeros_like(self.prev_obs)
        self.last_action = np.zeros(4, dtype=np.float32)
        self._finished = False
        self._step = 0
