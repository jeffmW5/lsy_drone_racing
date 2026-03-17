"""Gate-aware RL controller for Level 2 drone racing (exp_018).

Builds cubic spline trajectory from obs["gates_pos"] using the same approach
midpoint + gate waypoint pattern as training. Uses n_obs=2 policy trained on
gate-aware trajectories with L2 randomization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """RL controller with gate-aware trajectory generation."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.1
        self.trajectory_time = 15.0
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )
        self._tick = 0

        self.trajectory = None
        self._trajectory_built = False

        # Load RL policy (n_obs=2: 13 + 30 + 26 + 4 = 73)
        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
        model_path = Path(__file__).parent / "ppo_drone_racing_exp019.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.last_action = np.array([0.0, 0.0, 0.0, self.drone_mass * 9.81], dtype=np.float32)
        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))

        self._finished = False

    def _build_trajectory(self, gates_pos, drone_pos):
        """Build trajectory matching training: takeoff + climbout + (approach + gate) per gate."""
        # Use actual drone position as start (matches sim config, not training takeoff_pos)
        takeoff_pos = drone_pos.copy()
        takeoff_pos[2] = max(takeoff_pos[2], 0.05)  # ensure slightly above ground
        climbout = np.array([-1.0, 0.55, 0.4])

        # Build waypoints: same pattern as RandTrajEnv gate-aware reset
        n_gates = gates_pos.shape[0]
        waypoints = [takeoff_pos, climbout]

        prev_pos = climbout
        for i in range(n_gates):
            mid = (prev_pos + gates_pos[i]) / 2
            waypoints.append(mid)
            waypoints.append(gates_pos[i])
            prev_pos = gates_pos[i]

        waypoints = np.array(waypoints)
        num_wp = waypoints.shape[0]

        ts = np.linspace(0, self.trajectory_time, int(self.freq * self.trajectory_time))
        v0 = np.array([0.0, 0.0, 0.4])
        spline = CubicSpline(
            np.linspace(0, self.trajectory_time, num_wp),
            waypoints,
            bc_type=((1, v0), "not-a-knot"),
        )
        self.trajectory = spline(ts)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if not self._trajectory_built:
            self._build_trajectory(obs["gates_pos"], obs["pos"])
            self._trajectory_built = True

        i = min(self._tick, self.trajectory.shape[0] - 1)
        if i == self.trajectory.shape[0] - 1:
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_rl = torch.tensor(obs_rl, dtype=torch.float32).unsqueeze(0).to("cpu")
        with torch.no_grad():
            act, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)
            self.last_action = np.asarray(torch.asarray(act.squeeze(0))).copy()
            act[..., 2] = 0.0

        act = self._scale_actions(act.squeeze(0).numpy()).astype(np.float32)
        return act

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        obs_rl = {}
        obs_rl["basic_obs"] = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        dpos = self.trajectory[idx] - obs["pos"]
        obs_rl["local_samples"] = dpos.reshape(-1)
        obs_rl["prev_obs"] = self.prev_obs.reshape(-1)
        obs_rl["last_action"] = self.last_action
        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], obs_rl["basic_obs"][None, :]], axis=0
        )

        return np.concatenate([v for v in obs_rl.values()], axis=-1).astype(np.float32)

    def _scale_actions(self, actions: NDArray) -> NDArray:
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        mean = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0], dtype=np.float32
        )
        return np.clip(actions, -1.0, 1.0) * scale + mean

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._trajectory_built = False
