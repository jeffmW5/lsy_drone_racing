"""Training wrappers and env factory for VecDroneRaceEnv — direct gate-racing RL.

This replaces the trajectory-following pipeline (train_rl.py) with direct gate-racing
using VecDroneRaceEnv (MuJoCo physics, same as benchmark). Key differences:
- Dense reward based on gate proximity + passage bonus (not trajectory distance)
- Observations include relative gate positions (not trajectory sample points)
- Trains on the actual racing env, eliminating the crazyflow→MuJoCo physics gap
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.train_rl import ActionPenalty, FlattenJaxObservation
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from jax import Array


# =============================================================================
# Wrappers
# =============================================================================


class NormalizeRaceActions(VectorEnv):
    """Normalize agent actions from [-1, 1] to actual attitude action space.

    Also zeros out yaw command (index 2) for stability, matching the existing
    AngleReward wrapper behavior from train_rl.py.
    """

    def __init__(self, env: VecDroneRaceEnv):
        self.env = env
        self.num_envs = env.num_envs

        low = np.array(env.single_action_space.low, dtype=np.float32)
        high = np.array(env.single_action_space.high, dtype=np.float32)
        self._center = jp.array((high + low) / 2)
        self._scale = jp.array((high - low) / 2)

        self.single_action_space = spaces.Box(-1.0, 1.0, shape=low.shape, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Zero out yaw command
        if isinstance(action, np.ndarray):
            action = action.copy()
            action[..., 2] = 0.0
        else:
            action = action.at[..., 2].set(0.0)
        # Scale from [-1, 1] to actual attitude bounds
        scaled = np.asarray(action * self._scale + self._center)
        return self.env.step(scaled)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


class RaceRewardAndObs(VectorEnv):
    """Dense reward computation and observation preprocessing for gate racing.

    Reward components:
    - Gate proximity: exp(-proximity_coef * distance_to_target_gate)
    - Gate passage bonus: +gate_bonus per gate passed
    - Speed toward gate: speed_coef * max(velocity_toward_gate, 0)
    - RPY penalty: -rpy_coef * ||rpy||
    - Crash penalty: -1.0 on termination

    Observation: converts RaceCoreEnv dict obs to relative gate positions:
    - pos(3), quat(4), vel(3), ang_vel(3): drone state (13 dims)
    - rel_target_gate(3): relative position to current target gate
    - target_gate_quat(4): orientation of target gate
    - rel_next_gate(3): relative position to next gate (lookahead)
    - next_gate_quat(4): orientation of next gate
    Total: 30 dims (before stacking/action history)
    """

    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        gate_bonus: float = 5.0,
        proximity_coef: float = 2.0,
        speed_coef: float = 0.1,
        rpy_coef: float = 0.06,
        oob_coef: float = 0.0,
        z_low: float = 0.0,
        z_high: float = 2.0,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space

        self.n_gates = n_gates
        self.gate_bonus = gate_bonus
        self.proximity_coef = proximity_coef
        self.speed_coef = speed_coef
        self.rpy_coef = rpy_coef
        self.oob_coef = oob_coef
        self.z_low = z_low
        self.z_high = z_high

        # Define the preprocessed observation space
        obs_spec = {
            "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "quat": spaces.Box(-1, 1, shape=(4,)),
            "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "rel_target_gate": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "target_gate_quat": spaces.Box(-1, 1, shape=(4,)),
            "rel_next_gate": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "next_gate_quat": spaces.Box(-1, 1, shape=(4,)),
        }
        self.single_observation_space = spaces.Dict(obs_spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self._prev_target_gate = None

    def _preprocess_obs(self, obs: dict) -> dict:
        """Convert RaceCoreEnv dict obs to relative-position obs."""
        drone_pos = obs["pos"]  # (n_envs, 3)
        target_gate = obs["target_gate"]  # (n_envs,)
        gates_pos = obs["gates_pos"]  # (n_envs, n_gates, 3)
        gates_quat = obs["gates_quat"]  # (n_envs, n_gates, 4)

        # Clamp to valid range (crashed/completed drones use last valid gate)
        safe_target = jp.clip(target_gate, 0, self.n_gates - 1)
        next_target = jp.clip(target_gate + 1, 0, self.n_gates - 1)

        idx = jp.arange(self.num_envs)
        target_pos = gates_pos[idx, safe_target]
        target_quat = gates_quat[idx, safe_target]
        next_pos = gates_pos[idx, next_target]
        next_quat = gates_quat[idx, next_target]

        return {
            "pos": obs["pos"],
            "quat": obs["quat"],
            "vel": obs["vel"],
            "ang_vel": obs["ang_vel"],
            "rel_target_gate": target_pos - drone_pos,
            "target_gate_quat": target_quat,
            "rel_next_gate": next_pos - drone_pos,
            "next_gate_quat": next_quat,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_target_gate = jp.array(obs["target_gate"])
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        target_gate = obs["target_gate"]  # (n_envs,)
        gates_pos = obs["gates_pos"]  # (n_envs, n_gates, 3)
        drone_pos = obs["pos"]  # (n_envs, 3)
        drone_vel = obs["vel"]  # (n_envs, 3)
        drone_quat = obs["quat"]  # (n_envs, 4)

        # --- Gate proximity reward ---
        safe_target = jp.clip(target_gate, 0, self.n_gates - 1)
        idx = jp.arange(self.num_envs)
        target_pos = gates_pos[idx, safe_target]
        rel_pos = target_pos - drone_pos
        dist = jp.linalg.norm(rel_pos, axis=-1)
        proximity = jp.exp(-self.proximity_coef * dist)

        # --- Speed toward gate ---
        direction = rel_pos / (dist[:, None] + 1e-6)
        speed_toward = jp.sum(drone_vel * direction, axis=-1)
        speed_reward = self.speed_coef * jp.maximum(speed_toward, 0.0)

        # --- Gate passage bonus ---
        gate_passed = (target_gate > self._prev_target_gate) & (self._prev_target_gate >= 0)
        gate_reward = self.gate_bonus * gate_passed.astype(jp.float32)

        # --- RPY penalty ---
        rpy = R.from_quat(drone_quat).as_euler("xyz")
        rpy_penalty = self.rpy_coef * jp.linalg.norm(rpy, axis=-1)

        # --- Crash penalty ---
        crash_penalty = terminated.astype(jp.float32)

        # --- Out-of-bounds altitude penalty ---
        z = drone_pos[:, 2]
        oob_penalty = self.oob_coef * (
            jp.maximum(z - self.z_high, 0.0) + jp.maximum(self.z_low - z, 0.0)
        )

        # Only give proximity/speed reward to active (non-crashed) drones
        active = (target_gate >= 0).astype(jp.float32)
        reward = (
            active * (proximity + speed_reward)
            + gate_reward
            - crash_penalty
            - rpy_penalty
            - oob_penalty
        )

        self._prev_target_gate = jp.array(target_gate)

        return self._preprocess_obs(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


class RaceStackObs(VectorObservationWrapper):
    """Observation history stacking for RaceCoreEnv pipeline.

    Custom version of StackObs that doesn't call env.unwrapped.obs(),
    which would return unsqueezed obs from RaceCoreEnv (wrong shape).
    Initializes history buffer with zeros instead.
    """

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        super().__init__(env)
        self.n_obs = n_obs
        if self.n_obs > 0:
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(13 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 13))

    def observations(self, observations: dict) -> dict:
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations

    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        basic_obs_keys = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_keys], axis=-1
        )
        return jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)


# =============================================================================
# Environment Factory
# =============================================================================


def make_race_envs(
    config: str = "level2_attitude.toml",
    num_envs: int = 1024,
    jax_device: str = "cpu",
    torch_device=None,
    coefs: dict | None = None,
) -> VectorEnv:
    """Create VecDroneRaceEnv with training wrappers.

    Wrapper stack:
        VecDroneRaceEnv → NormalizeRaceActions → RaceRewardAndObs
        → RaceStackObs → ActionPenalty → FlattenJaxObservation → JaxToTorch
    """
    import torch

    if torch_device is None:
        torch_device = torch.device("cpu")
    if coefs is None:
        coefs = {}

    cfg = load_config(Path(__file__).parents[2] / "config" / config)
    control_mode = cfg.env.get("control_mode", "attitude")
    n_gates = len(cfg.env.track.gates)

    print(f"[make_race_envs] config={config}, num_envs={num_envs}, "
          f"control_mode={control_mode}, n_gates={n_gates}, device={jax_device}")

    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=control_mode,
        disturbances=cfg.env.get("disturbances", None),
        randomizations=cfg.env.get("randomizations", None),
        max_episode_steps=coefs.get("max_episode_steps", 1500),
        device=jax_device,
    )

    env = NormalizeRaceActions(env)
    env = RaceRewardAndObs(
        env,
        n_gates=n_gates,
        gate_bonus=coefs.get("gate_bonus", 5.0),
        proximity_coef=coefs.get("proximity_coef", 2.0),
        speed_coef=coefs.get("speed_coef", 0.1),
        rpy_coef=coefs.get("rpy_coef", 0.06),
        oob_coef=coefs.get("oob_coef", 0.0),
        z_low=coefs.get("z_low", 0.0),
        z_high=coefs.get("z_high", 2.0),
    )
    env = RaceStackObs(env, n_obs=coefs.get("n_obs", 2))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.02),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )
    env = FlattenJaxObservation(env)
    env = JaxToTorch(env, torch_device)

    print(f"[make_race_envs] obs_space={env.single_observation_space}, "
          f"act_space={env.single_action_space}")
    return env
