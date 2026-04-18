"""Training wrappers and env factory for VecDroneRaceEnv — direct gate-racing RL.

This replaces the trajectory-following pipeline (train_rl.py) with direct gate-racing
using VecDroneRaceEnv (MuJoCo physics, same as benchmark). Key differences:
- Dense reward based on gate proximity + passage bonus (not trajectory distance)
- Observations include relative gate positions (not trajectory sample points)
- Trains on the actual racing env, eliminating the crazyflow→MuJoCo physics gap
"""

from __future__ import annotations

from functools import partial
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
        action = jp.asarray(action)
        action = action.at[..., 2].set(0.0)
        # Scale from [-1, 1] to actual attitude bounds without host materialization.
        scaled = action * self._scale + self._center
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
        bilateral_progress: bool = False,
        body_frame_obs: bool = False,
        soft_collision: bool = False,
        soft_collision_penalty: float = 5.0,
        soft_collision_steps: int = 5_000_000,
        asymmetric_critic: bool = False,
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
        self.bilateral_progress = bilateral_progress
        self.body_frame_obs = body_frame_obs
        self.soft_collision = soft_collision
        self.soft_collision_penalty = soft_collision_penalty
        self.soft_collision_steps = soft_collision_steps
        self.asymmetric_critic = asymmetric_critic
        self._total_steps = 0
        self._last_privileged = None
        self._rng_key = jax.random.PRNGKey(0)

        # Define the preprocessed observation space
        if self.body_frame_obs:
            obs_spec = {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-1, 1, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "rel_target_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "target_normal_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "rel_next_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "next_normal_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        else:
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

    @staticmethod
    @jax.jit
    def _compute_prev_dist(pos, target_gate, gates_pos, n_gates: int):
        safe_t = jp.clip(target_gate, 0, n_gates - 1)
        target_pos = gates_pos[jp.arange(pos.shape[0]), safe_t]
        rel = target_pos - pos
        return jp.linalg.norm(rel[:, :2], axis=-1)

    @staticmethod
    @jax.jit
    def _preprocess_obs_world(obs: dict, n_gates: int) -> dict:
        drone_pos = obs["pos"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]

        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        next_target = jp.clip(target_gate + 1, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])

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

    @staticmethod
    @jax.jit
    def _preprocess_obs_body(obs: dict, n_gates: int) -> dict:
        drone_pos = obs["pos"]
        drone_quat = obs["quat"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]

        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        next_target = jp.clip(target_gate + 1, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])

        target_pos = gates_pos[idx, safe_target]
        target_quat = gates_quat[idx, safe_target]
        next_pos = gates_pos[idx, next_target]
        next_quat = gates_quat[idx, next_target]

        drone_rot_inv = R.from_quat(drone_quat).inv()
        rel_target_body = drone_rot_inv.apply(target_pos - drone_pos)
        rel_next_body = drone_rot_inv.apply(next_pos - drone_pos)

        unit_x = jp.broadcast_to(jp.array([1.0, 0.0, 0.0], dtype=drone_pos.dtype), drone_pos.shape)
        target_fwd = R.from_quat(target_quat).apply(unit_x)
        next_fwd = R.from_quat(next_quat).apply(unit_x)

        return {
            "pos": obs["pos"],
            "quat": obs["quat"],
            "vel": obs["vel"],
            "ang_vel": obs["ang_vel"],
            "rel_target_body": rel_target_body,
            "target_normal_body": drone_rot_inv.apply(target_fwd),
            "rel_next_body": rel_next_body,
            "next_normal_body": drone_rot_inv.apply(next_fwd),
        }

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("n_gates", "use_progress", "use_bilateral_progress", "use_view_multiply"),
    )
    def _compute_step_reward(
        *,
        obs: dict,
        prev_target_gate,
        prev_dist,
        n_gates: int,
        gate_bonus: float,
        proximity_coef: float,
        speed_coef: float,
        rpy_coef: float,
        oob_coef: float,
        z_low: float,
        z_high: float,
        alt_coef: float,
        survive_coef: float,
        vz_coef: float,
        vz_threshold: float,
        progress_coef: float,
        gate_in_view_coef: float,
        use_progress: bool,
        use_bilateral_progress: bool,
        use_view_multiply: bool,
        terminated,
    ):
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
        drone_pos = obs["pos"]
        drone_vel = obs["vel"]
        drone_quat = obs["quat"]

        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
        target_pos = gates_pos[idx, safe_target]
        rel_pos = target_pos - drone_pos
        dist = jp.linalg.norm(rel_pos, axis=-1)
        dist_xy = jp.linalg.norm(rel_pos[:, :2], axis=-1)

        if use_progress:
            delta = prev_dist - dist_xy
            if use_bilateral_progress:
                proximity = progress_coef * delta
            else:
                proximity = progress_coef * jp.maximum(delta, 0.0)
        else:
            proximity = jp.exp(-proximity_coef * dist)

        direction = rel_pos / (dist[:, None] + 1e-6)
        speed_toward = jp.sum(drone_vel * direction, axis=-1)
        speed_reward = speed_coef * jp.maximum(speed_toward, 0.0)

        gate_passed = (target_gate > prev_target_gate) & (prev_target_gate >= 0)
        gate_reward = gate_bonus * gate_passed.astype(jp.float32)

        rpy = R.from_quat(drone_quat).as_euler("xyz")
        rpy_penalty = rpy_coef * jp.linalg.norm(rpy, axis=-1)

        crash_penalty = terminated.astype(jp.float32)

        z = drone_pos[:, 2]
        oob_violation = (z > z_high) | (z < z_low)
        oob_penalty = oob_coef * (
            jp.maximum(z - z_high, 0.0) + jp.maximum(z_low - z, 0.0)
        )
        terminated = jp.where(oob_coef > 0, terminated | oob_violation, terminated)

        alt_error = jp.abs(z - target_pos[:, 2])
        alt_reward = alt_coef * jp.exp(-3.0 * alt_error)
        survive_reward = survive_coef

        vz_penalty = vz_coef * jp.maximum(drone_vel[:, 2], 0.0) * (
            z > vz_threshold
        ).astype(jp.float32)

        rot_mat = R.from_quat(drone_quat).as_matrix()
        drone_forward = rot_mat[:, :, 0]
        gate_dir = rel_pos / (dist[:, None] + 1e-6)
        alignment = jp.sum(drone_forward * gate_dir, axis=-1)
        view_reward = gate_in_view_coef * jp.maximum(alignment, 0.0)

        active = (target_gate >= 0).astype(jp.float32)
        additive_reward = active * (
            proximity + speed_reward + alt_reward + survive_reward + view_reward
        )
        multiplied_reward = active * (
            view_reward * proximity + speed_reward + alt_reward + survive_reward
        )
        reward = jp.where(use_view_multiply, multiplied_reward, additive_reward)
        reward = reward + gate_reward - crash_penalty - rpy_penalty - oob_penalty - vz_penalty

        return reward, terminated, target_gate, dist_xy

    def _preprocess_obs(self, obs: dict) -> dict:
        """Convert RaceCoreEnv dict obs to relative-position obs."""
        # Store privileged obs for asymmetric critic (all gate positions + quats)
        if self.asymmetric_critic:
            self._last_privileged = jp.concatenate([
                obs["gates_pos"].reshape(self.num_envs, -1),   # (n_envs, n_gates*3)
                obs["gates_quat"].reshape(self.num_envs, -1),  # (n_envs, n_gates*4)
            ], axis=-1)

        if self.body_frame_obs:
            return self._preprocess_obs_body(obs, self.n_gates)
        return self._preprocess_obs_world(obs, self.n_gates)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(int(seed))
        obs, info = self.env.reset(**kwargs)
        if self.random_gate_start:
            obs = self._apply_random_gate_start(obs)
        self._prev_target_gate = jp.array(obs["target_gate"])
        self._was_done = None
        # Initialize prev_dist for delta-progress reward (XY only — ignore altitude)
        if self.progress_coef > 0.0:
            self._prev_dist = self._compute_prev_dist(
                jp.array(obs["pos"]),
                jp.array(obs["target_gate"]),
                jp.array(obs["gates_pos"]),
                self.n_gates,
            )
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += self.num_envs

        # Save real termination for autoreset tracking (before soft-collision suppression)
        real_terminated = terminated

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
                # Sync _prev_dist for respawned envs (prevents spurious progress delta)
                if self.progress_coef > 0.0 and self._prev_dist is not None:
                    new_dist = self._compute_prev_dist(
                        jp.array(obs["pos"]),
                        new_target,
                        jp.array(obs["gates_pos"]),
                        self.n_gates,
                    )
                    self._prev_dist = jp.where(
                        jp.array(autoreset_mask), new_dist, self._prev_dist
                    )
        reward, terminated, target_gate, dist_xy = self._compute_step_reward(
            obs=obs,
            prev_target_gate=self._prev_target_gate,
            prev_dist=self._prev_dist,
            n_gates=self.n_gates,
            gate_bonus=self.gate_bonus,
            proximity_coef=self.proximity_coef,
            speed_coef=self.speed_coef,
            rpy_coef=self.rpy_coef,
            oob_coef=self.oob_coef,
            z_low=self.z_low,
            z_high=self.z_high,
            alt_coef=self.alt_coef,
            survive_coef=self.survive_coef,
            vz_coef=self.vz_coef,
            vz_threshold=self.vz_threshold,
            progress_coef=self.progress_coef,
            gate_in_view_coef=self.gate_in_view_coef,
            use_progress=self.progress_coef > 0.0 and self._prev_dist is not None,
            use_bilateral_progress=self.bilateral_progress,
            use_view_multiply=self.reward_mode == "multiply" and self.gate_in_view_coef > 0.0,
            terminated=terminated,
        )

        # === SOFT COLLISION: suppress termination during phase 1 ===
        if self.soft_collision and self._total_steps < self.soft_collision_steps:
            soft_crashed = np.array(real_terminated) & ~np.array(truncated)
            if np.any(soft_crashed):
                sc = jp.array(soft_crashed)
                # Override reward for soft-crashed envs (replace all components with flat penalty)
                reward = jp.where(sc, jp.float32(-self.soft_collision_penalty), reward)
                # Suppress termination signal — PPO sees episode continuing
                terminated = jp.where(sc, False, terminated)
            # Log phase transition
            if (self._total_steps - self.num_envs < self.soft_collision_steps
                    <= self._total_steps):
                print(f"[SOFT COLLISION] Phase 2 at step {self._total_steps}: "
                      f"hard termination on crash")

        self._prev_target_gate = jp.array(target_gate)
        if self.progress_coef > 0.0:
            self._prev_dist = dist_xy
        self._was_done = np.array(real_terminated | truncated)

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
        self._rng_key, mask_key = jax.random.split(self._rng_key)
        random_mask = jax.random.uniform(mask_key, (self.num_envs,)) < self.random_gate_ratio
        if mask is None:
            mask = random_mask
        else:
            mask = jp.asarray(mask) & random_mask

        if not bool(jp.any(mask)):
            return obs

        self._rng_key, gate_key, pos_key, vel_key, angle_key = jax.random.split(
            self._rng_key, 5
        )

        target_gate = jp.asarray(obs["target_gate"], dtype=jp.int32)
        gates_pos = jp.asarray(obs["gates_pos"])
        gates_quat = jp.asarray(obs["gates_quat"])

        random_gates = jax.random.randint(gate_key, (self.num_envs,), 0, self.n_gates)
        spawn_gates_full = jp.where(mask, random_gates, target_gate)

        idx = jp.arange(self.num_envs)
        target_pos = gates_pos[idx, spawn_gates_full]
        target_quat = gates_quat[idx, spawn_gates_full]

        # Gate forward = local x-axis (gates are crossed -x → +x)
        gate_rot = R.from_quat(target_quat)
        unit_x = jp.broadcast_to(jp.array([1.0, 0.0, 0.0], dtype=target_pos.dtype), target_pos.shape)
        forward = gate_rot.apply(unit_x)

        # Spawn position: offset before gate + noise
        spawn_pos = target_pos - self.spawn_offset * forward
        pos_noise = jax.random.uniform(
            pos_key,
            (self.num_envs, 3),
            minval=-self.spawn_pos_noise,
            maxval=self.spawn_pos_noise,
        )
        spawn_pos += pos_noise

        # Drone orientation: face the gate (match gate yaw) + perturbation
        gate_euler = gate_rot.as_euler("xyz")
        angle_min = jp.array(
            [-np.radians(5), -np.radians(5), -np.radians(15)], dtype=target_pos.dtype
        )
        angle_max = jp.array(
            [np.radians(5), np.radians(5), np.radians(15)], dtype=target_pos.dtype
        )
        angle_noise = jax.random.uniform(
            angle_key,
            (self.num_envs, 3),
            minval=angle_min,
            maxval=angle_max,
        )
        drone_rpy = angle_noise.at[:, 2].add(gate_euler[:, 2])
        drone_quat = R.from_euler("xyz", drone_rpy).as_quat()

        # Spawn velocity: small random
        spawn_vel = jax.random.uniform(
            vel_key,
            (self.num_envs, 3),
            minval=-self.spawn_vel_noise,
            maxval=self.spawn_vel_noise,
        )

        # Merge with existing state for non-masked envs
        core_env = self.env.unwrapped
        old_pos = core_env.sim.data.states.pos[:, 0]
        old_quat = core_env.sim.data.states.quat[:, 0]
        old_vel = core_env.sim.data.states.vel[:, 0]

        new_pos = jp.where(mask[:, None], spawn_pos, old_pos).astype(jp.float32)
        new_quat = jp.where(mask[:, None], drone_quat, old_quat).astype(jp.float32)
        new_vel = jp.where(mask[:, None], spawn_vel, old_vel).astype(jp.float32)

        # Override sim state
        pos = core_env.sim.data.states.pos.at[:, 0, :].set(new_pos)
        quat = core_env.sim.data.states.quat.at[:, 0, :].set(new_quat)
        vel = core_env.sim.data.states.vel.at[:, 0, :].set(new_vel)
        ang_vel = core_env.sim.data.states.ang_vel.at[:, 0, :].set(
            jp.zeros((self.num_envs, 3), dtype=jp.float32)
        )
        core_env.sim.data = core_env.sim.data.replace(
            states=core_env.sim.data.states.replace(
                pos=pos, quat=quat, vel=vel, ang_vel=ang_vel
            )
        )

        # Override target gate and last_drone_pos
        old_target = core_env.data.target_gate[:, 0]
        new_target = jp.where(mask, spawn_gates_full, old_target).astype(jp.int32)
        core_env.data = core_env.data.replace(
            target_gate=new_target[:, None],
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


class AppendPrivilegedObs(VectorEnv):
    """Append privileged observations for asymmetric actor-critic (exp_059).

    Adds all gate positions and quaternions to the observation dict so the critic
    can see the full track layout. Placed after ActionPenalty so privileged dims
    are at the END of the flat vector. Actor uses x[:, :actor_obs_dim], critic
    uses full x.
    """

    def __init__(self, env: VectorEnv, reward_wrapper: RaceRewardAndObs):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self._reward_wrapper = reward_wrapper

        # Compute actor obs dim (flat dim of all keys before privileged)
        self.actor_obs_dim = sum(
            int(np.prod(v.shape)) for v in env.single_observation_space.values()
        )

        # Add privileged obs to observation space
        n_gates = reward_wrapper.n_gates
        self._priv_dim = n_gates * 3 + n_gates * 4  # positions + quaternions
        spec = dict(env.single_observation_space.items())
        spec["privileged_obs"] = spaces.Box(-np.inf, np.inf, shape=(self._priv_dim,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _add_privileged(self, obs: dict) -> dict:
        priv = self._reward_wrapper._last_privileged
        if priv is None:
            priv = jp.zeros((self.num_envs, self._priv_dim))
        obs["privileged_obs"] = priv
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_privileged(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_privileged(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


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
        bilateral_progress=coefs.get("bilateral_progress", False),
        body_frame_obs=coefs.get("body_frame_obs", False),
        soft_collision=coefs.get("soft_collision", False),
        soft_collision_penalty=coefs.get("soft_collision_penalty", 5.0),
        soft_collision_steps=coefs.get("soft_collision_steps", 5_000_000),
        asymmetric_critic=coefs.get("asymmetric_critic", False),
    )
    reward_wrapper = env  # keep reference for AppendPrivilegedObs

    env = RaceStackObs(env, n_obs=coefs.get("n_obs", 2))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.02),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )

    # Asymmetric critic: append privileged obs (all gate positions/quats) at end
    if coefs.get("asymmetric_critic", False):
        env = AppendPrivilegedObs(env, reward_wrapper)
        print(f"[make_race_envs] asymmetric_critic: privileged_dim={n_gates * 7}, "
              f"actor_obs_dim={env.actor_obs_dim}")

    env = FlattenJaxObservation(env)
    env = JaxToTorch(env, torch_device)

    # Propagate actor_obs_dim to the final env for train_racing.py to read
    if coefs.get("asymmetric_critic", False):
        env.actor_obs_dim = int(np.prod(env.single_observation_space.shape)) - n_gates * 7

    print(f"[make_race_envs] obs_space={env.single_observation_space}, "
          f"act_space={env.single_action_space}")
    return env
