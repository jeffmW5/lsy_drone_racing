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
from scipy.spatial.transform import Rotation as ScipyR

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
        alt_coef: float = 0.0,
        survive_coef: float = 0.0,
        vz_coef: float = 0.0,
        vz_threshold: float = 0.5,
        random_gate_start: bool = False,
        random_gate_ratio: float = 1.0,
        progress_coef: float = 0.0,
        gate_in_view_coef: float = 0.0,
        reward_mode: str = "add",  # "add" or "multiply" (view * progress)
        spawn_offset: float = 0.75,
        spawn_pos_noise: float = 0.15,
        spawn_vel_noise: float = 0.3,
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
        self.alt_coef = alt_coef
        self.survive_coef = survive_coef
        self.vz_coef = vz_coef
        self.vz_threshold = vz_threshold
        self.random_gate_start = random_gate_start
        self.random_gate_ratio = random_gate_ratio  # fraction of envs that get random gate start
        self.spawn_offset = spawn_offset
        self.spawn_pos_noise = spawn_pos_noise
        self.spawn_vel_noise = spawn_vel_noise
        self._rng = np.random.default_rng()

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
        self._prev_dist = None
        self.progress_coef = progress_coef
        self.gate_in_view_coef = gate_in_view_coef
        self.reward_mode = reward_mode
        self._was_done = None  # Track terminated|truncated for autoreset detection

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
        if self.random_gate_start:
            obs = self._apply_random_gate_start(obs)
        self._prev_target_gate = jp.array(obs["target_gate"])
        self._was_done = None
        # Initialize prev_dist for delta-progress reward (XY only — ignore altitude)
        if self.progress_coef > 0.0:
            safe_t = jp.clip(jp.array(obs["target_gate"]), 0, self.n_gates - 1)
            idx0 = jp.arange(self.num_envs)
            t_pos = jp.array(obs["gates_pos"])[idx0, safe_t]
            rel = t_pos - jp.array(obs["pos"])
            self._prev_dist = jp.linalg.norm(rel[:, :2], axis=-1)
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        # Apply random gate start to autoreset envs
        if self.random_gate_start and self._was_done is not None:
            autoreset_mask = np.array(self._was_done)
            if autoreset_mask.any():
                obs = self._apply_random_gate_start(obs, mask=autoreset_mask)
                # Sync _prev_target_gate to prevent false gate bonuses
                new_target = jp.array(obs["target_gate"])
                self._prev_target_gate = jp.where(
                    jp.array(autoreset_mask), new_target, self._prev_target_gate
                )

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
        # XY-only distance for progress reward — prevents gaming by falling
        dist_xy = jp.linalg.norm(rel_pos[:, :2], axis=-1)
        if self.progress_coef > 0.0 and self._prev_dist is not None:
            # Potential-based reward shaping using horizontal distance only
            proximity = self.progress_coef * jp.maximum(self._prev_dist - dist_xy, 0.0)
        else:
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

        # --- Out-of-bounds altitude penalty + hard termination ---
        z = drone_pos[:, 2]
        oob_violation = (z > self.z_high) | (z < self.z_low)
        oob_penalty = self.oob_coef * (
            jp.maximum(z - self.z_high, 0.0) + jp.maximum(self.z_low - z, 0.0)
        )
        # Hard-terminate OOB drones (overrides grace period)
        if self.oob_coef > 0:
            terminated = terminated | oob_violation

        # --- Altitude-matching reward: reward being at target gate's z ---
        target_z = target_pos[:, 2]
        alt_error = jp.abs(z - target_z)
        alt_reward = self.alt_coef * jp.exp(-3.0 * alt_error)

        # --- Survival bonus: reward for staying alive ---
        survive_reward = self.survive_coef

        # --- Vertical velocity penalty: penalize upward velocity when above threshold ---
        vz = drone_vel[:, 2]
        vz_penalty = self.vz_coef * jp.maximum(vz, 0.0) * (z > self.vz_threshold).astype(jp.float32)

        # --- Gate-in-view reward: alignment of drone forward axis with direction to gate ---
        if self.gate_in_view_coef > 0.0:
            rot_mat = R.from_quat(drone_quat).as_matrix()  # (n_envs, 3, 3)
            drone_forward = rot_mat[:, :, 0]  # body x-axis in world frame
            gate_dir = rel_pos / (dist[:, None] + 1e-6)
            alignment = jp.sum(drone_forward * gate_dir, axis=-1)  # cosine [-1, 1]
            view_reward = self.gate_in_view_coef * jp.maximum(alignment, 0.0)
        else:
            view_reward = 0.0

        # Only give proximity/speed/alt reward to active (non-crashed) drones
        active = (target_gate >= 0).astype(jp.float32)
        if self.reward_mode == "multiply" and self.gate_in_view_coef > 0.0:
            # view * progress: zero reward unless both facing AND moving
            reward = active * (view_reward * proximity + speed_reward + alt_reward + survive_reward)
        else:
            reward = active * (proximity + speed_reward + alt_reward + survive_reward + view_reward)
        reward = (
            reward
            + gate_reward
            - crash_penalty
            - rpy_penalty
            - oob_penalty
            - vz_penalty
        )

        self._prev_target_gate = jp.array(target_gate)
        if self.progress_coef > 0.0:
            self._prev_dist = dist_xy
        self._was_done = np.array(terminated | truncated)

        return self._preprocess_obs(obs), reward, terminated, truncated, info

    def _apply_random_gate_start(self, obs: dict, mask: np.ndarray | None = None) -> dict:
        """Override drone spawn to random gate positions.

        After a normal reset, moves drones to a random position before a random gate,
        facing toward it, with small perturbation in position/velocity/orientation.

        Args:
            obs: Observation dict from VecDroneRaceEnv (with drone dim squeezed).
            mask: Boolean array (n_envs,) — only modify envs where True.
                  If None, apply to random_gate_ratio fraction of envs.
        """
        if mask is None:
            # Initial reset: apply to random_gate_ratio fraction
            mask = self._rng.random(self.num_envs) < self.random_gate_ratio
        else:
            # Autoreset: apply ratio within the autoreset mask
            mask = mask & (self._rng.random(self.num_envs) < self.random_gate_ratio)
        n_reset = int(mask.sum())
        if n_reset == 0:
            return obs

        # Pick random gates for reset envs
        spawn_gates_full = np.array(obs["target_gate"], dtype=int)
        spawn_gates_full[mask] = self._rng.integers(0, self.n_gates, size=n_reset)

        gates_pos = np.array(obs["gates_pos"])  # (n_envs, n_gates, 3)
        gates_quat = np.array(obs["gates_quat"])  # (n_envs, n_gates, 4) scipy order

        idx = np.arange(self.num_envs)
        target_pos = gates_pos[idx, spawn_gates_full]  # (n_envs, 3)
        target_quat = gates_quat[idx, spawn_gates_full]  # (n_envs, 4)

        # Gate forward = local x-axis (gates are crossed -x → +x)
        gate_rot = ScipyR.from_quat(target_quat)
        forward = gate_rot.apply(np.array([1.0, 0.0, 0.0]))  # (n_envs, 3)

        # Spawn position: offset before gate + noise
        spawn_pos = target_pos - self.spawn_offset * forward
        pos_noise = self._rng.uniform(
            -self.spawn_pos_noise, self.spawn_pos_noise, size=(self.num_envs, 3)
        )
        spawn_pos += pos_noise

        # Drone orientation: face the gate (match gate yaw) + perturbation
        gate_euler = gate_rot.as_euler("xyz")  # (n_envs, 3)
        drone_rpy = np.zeros((self.num_envs, 3), dtype=np.float64)
        drone_rpy[:, 2] = gate_euler[:, 2]  # Match gate yaw
        drone_rpy[:, 0] += self._rng.uniform(-np.radians(5), np.radians(5), size=self.num_envs)
        drone_rpy[:, 1] += self._rng.uniform(-np.radians(5), np.radians(5), size=self.num_envs)
        drone_rpy[:, 2] += self._rng.uniform(-np.radians(15), np.radians(15), size=self.num_envs)
        drone_quat = ScipyR.from_euler("xyz", drone_rpy).as_quat()  # (n_envs, 4)

        # Spawn velocity: small random
        spawn_vel = self._rng.uniform(
            -self.spawn_vel_noise, self.spawn_vel_noise, size=(self.num_envs, 3)
        )

        # Merge with existing state for non-masked envs
        core_env = self.env.unwrapped
        old_pos = np.array(core_env.sim.data.states.pos[:, 0])
        old_quat = np.array(core_env.sim.data.states.quat[:, 0])
        old_vel = np.array(core_env.sim.data.states.vel[:, 0])

        new_pos = np.where(mask[:, None], spawn_pos, old_pos).astype(np.float32)
        new_quat = np.where(mask[:, None], drone_quat, old_quat).astype(np.float32)
        new_vel = np.where(mask[:, None], spawn_vel, old_vel).astype(np.float32)

        # Override sim state
        pos = core_env.sim.data.states.pos.at[:, 0, :].set(new_pos)
        quat = core_env.sim.data.states.quat.at[:, 0, :].set(new_quat)
        vel = core_env.sim.data.states.vel.at[:, 0, :].set(new_vel)
        ang_vel = core_env.sim.data.states.ang_vel.at[:, 0, :].set(
            np.zeros((self.num_envs, 3), dtype=np.float32)
        )
        core_env.sim.data = core_env.sim.data.replace(
            states=core_env.sim.data.states.replace(
                pos=pos, quat=quat, vel=vel, ang_vel=ang_vel
            )
        )

        # Override target gate and last_drone_pos
        old_target = np.array(core_env.data.target_gate[:, 0])
        new_target = np.where(mask, spawn_gates_full, old_target).astype(int)
        core_env.data = core_env.data.replace(
            target_gate=jp.array(new_target[:, None]),
            last_drone_pos=pos,
        )

        # Update obs dict with new state
        obs = dict(obs)  # shallow copy to avoid mutating caller's dict
        obs["pos"] = core_env.sim.data.states.pos[:, 0]
        obs["quat"] = core_env.sim.data.states.quat[:, 0]
        obs["vel"] = core_env.sim.data.states.vel[:, 0]
        obs["ang_vel"] = core_env.sim.data.states.ang_vel[:, 0]
        obs["target_gate"] = core_env.data.target_gate[:, 0]

        return obs

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
        alt_coef=coefs.get("alt_coef", 0.0),
        survive_coef=coefs.get("survive_coef", 0.0),
        vz_coef=coefs.get("vz_coef", 0.0),
        vz_threshold=coefs.get("vz_threshold", 0.5),
        random_gate_start=coefs.get("random_gate_start", False),
        random_gate_ratio=coefs.get("random_gate_ratio", 1.0),
        progress_coef=coefs.get("progress_coef", 0.0),
        gate_in_view_coef=coefs.get("gate_in_view_coef", 0.0),
        reward_mode=coefs.get("reward_mode", "add"),
        spawn_offset=coefs.get("spawn_offset", 0.75),
        spawn_pos_noise=coefs.get("spawn_pos_noise", 0.15),
        spawn_vel_noise=coefs.get("spawn_vel_noise", 0.3),
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
