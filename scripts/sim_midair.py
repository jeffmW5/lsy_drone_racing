"""Benchmark sim that starts the drone mid-air, flying toward gate 0.

Eliminates the ground takeoff — matches the training spawn conditions from
experiments using random_gate_start. The drone starts 0.75m before gate 0,
at gate altitude, facing the gate, with zero velocity.

Usage:
    DRONE_RL_CKPT_PATH=/path/to/model.ckpt python scripts/sim_midair.py \
        --config level2_attitude.toml --controller attitude_rl_race.py --n_runs 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as ScipyR

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller

logger = logging.getLogger(__name__)


def override_to_midair(env, obs, gate_idx=0, offset=0.75):
    """Override drone state to start mid-air before a gate.

    Args:
        env: The JaxToNumpy-wrapped DroneRaceEnv.
        obs: The observation dict from reset().
        gate_idx: Which gate to spawn before.
        offset: Distance before the gate to spawn.

    Returns:
        Updated obs dict with new drone state.
    """
    core = env.unwrapped
    gates_pos = np.array(obs["gates_pos"])
    gates_quat = np.array(obs["gates_quat"])

    target_pos = gates_pos[gate_idx]
    target_quat = gates_quat[gate_idx]

    # Gate forward = local x-axis (gates are crossed -x -> +x)
    gate_rot = ScipyR.from_quat(target_quat)
    forward = gate_rot.apply(np.array([1.0, 0.0, 0.0]))

    # Spawn position: offset before gate, at gate altitude
    spawn_pos = target_pos - offset * forward

    # Drone orientation: face the gate (match gate yaw)
    gate_euler = gate_rot.as_euler("xyz")
    drone_rpy = np.array([0.0, 0.0, gate_euler[2]])
    drone_quat = ScipyR.from_euler("xyz", drone_rpy).as_quat().astype(np.float32)

    # Zero velocity
    spawn_vel = np.zeros(3, dtype=np.float32)
    spawn_ang_vel = np.zeros(3, dtype=np.float32)

    # Override sim state (single env, drone dim = [:, 0, :])
    new_pos = jp.array(spawn_pos[None, None, :].astype(np.float32))
    new_quat = jp.array(drone_quat[None, None, :].astype(np.float32))
    new_vel = jp.array(spawn_vel[None, None, :].astype(np.float32))
    new_ang_vel = jp.array(spawn_ang_vel[None, None, :].astype(np.float32))

    core.sim.data = core.sim.data.replace(
        states=core.sim.data.states.replace(
            pos=new_pos, quat=new_quat, vel=new_vel, ang_vel=new_ang_vel
        )
    )

    # Update obs to reflect new state
    obs = dict(obs)
    obs["pos"] = np.array(spawn_pos)
    obs["quat"] = np.array(drone_quat)
    obs["vel"] = np.array(spawn_vel)
    obs["ang_vel"] = np.array(spawn_ang_vel)

    return obs


def simulate(
    config: str = "level2_attitude.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller with mid-air start."""
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for run in range(n_runs):
        obs, info = env.reset()
        # Override to mid-air start before gate 0
        obs = override_to_midair(env, obs, gate_idx=0, offset=0.75)

        controller_inst: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq
            action = controller_inst.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller_inst.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
            i += 1

        controller_inst.episode_callback()
        gates_passed = obs["target_gate"]
        if gates_passed == -1:
            gates_passed = len(config.env.track.gates)
        finished = gates_passed == len(config.env.track.gates)
        logger.info(
            f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
        )
        controller_inst.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
