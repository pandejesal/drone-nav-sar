#!/usr/bin/env python3
"""
Shared mock physics backend for DroneNav-SAR sim envs.

Used automatically when the real simulator for a backend is unavailable
(no Isaac Sim install, no AirSim/Unreal endpoint, no ROS2/Gazebo daemon).
Integrates the local QuadrotorDynamics model, applies DomainRandomizer
scales, and synthesizes headless RGB/depth frames.

Every backend env delegates reset/step/render here in mock mode, so the
Gymnasium interface, reward shaping and randomization are identical
across backends and testable without GPU/simulator installs.

Mock mode is explicit: info["mock_backend"] is True and each env exposes
.use_mock. Real-backend hooks live in the per-backend env modules.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_env import DroneState, SimConfig
from .drone_dynamics import QuadrotorDynamics, QuadrotorParams
from .domain_randomization import DomainRandomizer, get_dynamic_config


@dataclass
class DynamicObstacle:
    """Sprint 5 moving obstacle: wandering person or cycling door.

    People are vertical cylinders (radius 0.3 m, height 1.8 m) doing a
    random walk at 0.5-1.0 m/s, bouncing off the room walls. Doors are
    1.0x2.1 m panels running an open/close cycle (period 10-30 s) that
    block passage while closed.
    """

    PERSON_RADIUS = 0.3
    PERSON_HEIGHT = 1.8
    DOOR_WIDTH = 1.0
    DOOR_HEIGHT = 2.1

    def __init__(
        self,
        kind: str = "person",
        position: Optional[np.ndarray] = None,
        speed: float = 0.75,
        period: float = 20.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.kind = kind
        self.position = np.array(
            position if position is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.speed = float(speed)
        self.period = float(period)
        self._rng = rng
        heading = float((rng or np.random.default_rng()).uniform(0, 2 * np.pi))
        self.velocity = np.array(
            [np.cos(heading) * self.speed, np.sin(heading) * self.speed, 0.0],
            dtype=np.float32,
        )
        self.phase = float((rng or np.random.default_rng()).uniform(0, period))
        self.open_fraction = 1.0 if kind == "door" else 0.0

    def update(
        self,
        dt: float,
        rng: Optional[np.random.Generator] = None,
        bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
    ) -> None:
        """Advance one obstacle step (random walk / door cycle)."""
        rng = rng or self._rng or np.random.default_rng()
        if self.kind == "person":
            # Random-walk heading jitter, then bounce off walls.
            self.phase += dt
            heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
            heading += float(rng.normal(0.0, 0.8 * dt))
            mag = self.speed
            self.velocity = np.array(
                [np.cos(heading) * mag, np.sin(heading) * mag, 0.0],
                dtype=np.float32,
            )
            self.position[:2] += self.velocity[:2] * dt
            x_min, x_max, y_min, y_max = bounds
            if self.position[0] < x_min or self.position[0] > x_max:
                self.velocity[0] *= -1.0
                self.position[0] = float(np.clip(self.position[0], x_min, x_max))
            if self.position[1] < y_min or self.position[1] > y_max:
                self.velocity[1] *= -1.0
                self.position[1] = float(np.clip(self.position[1], y_min, y_max))
        elif self.kind == "door":
            # Open/close cycle: blocked while closed (first half-period).
            self.phase = (self.phase + dt) % self.period
            self.open_fraction = 0.5 - 0.5 * float(
                np.cos(2.0 * np.pi * self.phase / self.period)
            )

    @property
    def blocked(self) -> bool:
        """Doors block passage while closed; people always collide."""
        if self.kind == "door":
            return self.open_fraction < 0.5
        return True

    def check_collision(
        self, drone_pos: np.ndarray, drone_radius: float = 0.2
    ) -> bool:
        """True when the drone body intersects this obstacle."""
        drone_pos = np.asarray(drone_pos, dtype=np.float32)
        if self.kind == "person":
            if float(drone_pos[2]) > self.PERSON_HEIGHT:
                return False
            dist2d = float(np.linalg.norm(drone_pos[:2] - self.position[:2]))
            return dist2d < (self.PERSON_RADIUS + drone_radius)
        # Door panel: blocked only while closed.
        if not self.blocked:
            return False
        if float(drone_pos[2]) > self.DOOR_HEIGHT:
            return False
        dist2d = float(np.linalg.norm(drone_pos[:2] - self.position[:2]))
        return dist2d < (self.DOOR_WIDTH / 2.0 + drone_radius)


@dataclass
class Victim:
    """Sprint 7: static victim entity for SAR mission.

    Victims are stationary cylinders (radius 0.3 m, height 1.8 m) that
    need med-kit delivery. They don't move but can be detected by YOLO.
    """

    VICTIM_RADIUS = 0.3
    VICTIM_HEIGHT = 1.8

    def __init__(
        self,
        position: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ):
        self.position = np.array(position, dtype=np.float32)
        self._rng = rng
        self.served = False

    def check_collision(
        self, drone_pos: np.ndarray, drone_radius: float = 0.2
    ) -> bool:
        """Check if drone is close enough to drop med-kit."""
        drone_pos = np.asarray(drone_pos, dtype=np.float32)
        if float(drone_pos[2]) > self.VICTIM_HEIGHT:
            return False
        dist2d = float(np.linalg.norm(drone_pos[:2] - self.position[:2]))
        return dist2d < (self.VICTIM_RADIUS + drone_radius)

    def serve(self) -> None:
        """Mark victim as served (med-kit delivered)."""
        self.served = True


class MockDroneBackend:
    """Headless quadrotor backend: dynamics + wind + synthetic camera + dynamic obstacles + SAR mission."""

    # Curriculum reward schedules (level 0=easy .. 3=target spec)
    CURRICULUM = [
        {"goal_tolerance": 1.5, "dist_w": -0.3, "vel_w": 0.5, "effort_w": -0.01, "alive_w": 0.10, "max_steps": 300},
        {"goal_tolerance": 1.0, "dist_w": -0.4, "vel_w": 0.4, "effort_w": -0.015, "alive_w": 0.07, "max_steps": 400},
        {"goal_tolerance": 0.7, "dist_w": -0.5, "vel_w": 0.3, "effort_w": -0.02, "alive_w": 0.05, "max_steps": 500},
        {"goal_tolerance": 0.5, "dist_w": -0.5, "vel_w": 0.3, "effort_w": -0.02, "alive_w": 0.05, "max_steps": 500},
    ]

    COLLISION_PENALTY = 10.0
    ROOM_BOUNDS = (-5.0, 5.0, -5.0, 5.0)

    def __init__(
        self,
        config: SimConfig,
        backend_name: str = "mock",
        rng: Optional[np.random.Generator] = None,
        curriculum_level: int = 3,
        dynamic_obstacles: bool = False,
        num_people: int = 0,
        use_doors: bool = False,
        # Sprint 7: SAR mission settings
        sar_mission: bool = False,
        victims: Optional[List[np.ndarray]] = None,
    ):
        self.config = config
        self.backend_name = backend_name
        self._rng = rng or np.random.default_rng()

        self._nominal_mass = float(config.mass)
        self._nominal_thrust = float(config.max_thrust)
        self._nominal_tau = 0.02

        params = QuadrotorParams(
            mass=config.mass,
            max_thrust=config.max_thrust,
            max_rpm=config.max_rpm,
            motor_time_constant=self._nominal_tau,
        )
        self.dynamics = QuadrotorDynamics(params)
        self.randomizer = DomainRandomizer(config)
        self.randomizer._rng = self._rng
        self._state_vec: Optional[np.ndarray] = None
        self.dr_params: Dict = {}
        self.curriculum_level = curriculum_level
        # Sprint 5 dynamic-obstacle settings (explicit counts win over the
        # curriculum row; reset() spawns from the resolved config).
        self.dynamic_obstacles = bool(dynamic_obstacles)
        self.num_people = int(num_people)
        self.use_doors = bool(use_doors)
        self.obstacles: list = []

        # Sprint 7: SAR mission
        self.sar_mission = bool(sar_mission)
        self.victims: list = []
        self.medkit_carried = True
        self.victims_served = 0
        self.total_victims = 0

    @property
    def curriculum_level(self) -> int:
        return self._curriculum_level

    @curriculum_level.setter
    def curriculum_level(self, level: int):
        self._curriculum_level = max(0, min(level, len(self.CURRICULUM) - 1))

    def _curriculum_params(self) -> Dict:
        return self.CURRICULUM[self._curriculum_level]

    @property
    def goal_tolerance(self) -> float:
        """Goal radius (m) for the current curriculum level."""
        return float(self._curriculum_params()["goal_tolerance"])

    @property
    def max_steps(self) -> int:
        """Episode step budget for the current curriculum level."""
        return int(self._curriculum_params()["max_steps"])

    # -- dynamic obstacles (Sprint 5) ------------------------------------

    def _resolved_dynamic_config(self) -> Dict:
        """Resolve (num_people, use_doors) from explicit counts or curriculum."""
        if self.num_people > 0 or self.use_doors:
            return {"num_people": self.num_people, "use_doors": self.use_doors}
        if not self.dynamic_obstacles:
            return {"num_people": 0, "use_doors": False}
        return get_dynamic_config(self._curriculum_level)

    def spawn_obstacles(
        self,
        n_people: Optional[int] = None,
        use_doors: Optional[bool] = None,
    ) -> list:
        """Spawn moving obstacles (people + doors). Returns the obstacle list."""
        if n_people is None or use_doors is None:
            resolved = self._resolved_dynamic_config()
            if n_people is None:
                n_people = resolved["num_people"]
            if use_doors is None:
                use_doors = resolved["use_doors"]
        x_min, x_max, y_min, y_max = self.ROOM_BOUNDS
        self.obstacles = []
        for _ in range(int(n_people or 0)):
            pos = np.array([
                float(self._rng.uniform(x_min, x_max)),
                float(self._rng.uniform(y_min, y_max)),
                0.0,
            ], dtype=np.float32)
            speed = float(self._rng.uniform(0.5, 1.0))
            self.obstacles.append(DynamicObstacle(
                kind="person", position=pos, speed=speed, rng=self._rng))
        if bool(use_doors):
            pos = np.array([
                float(self._rng.uniform(x_min, x_max)),
                float(self._rng.uniform(y_min, y_max)),
                0.0,
            ], dtype=np.float32)
            period = float(self._rng.uniform(10.0, 30.0))
            self.obstacles.append(DynamicObstacle(
                kind="door", position=pos, period=period, rng=self._rng))
        return self.obstacles

    def step_obstacles(self, dt: float) -> None:
        """Advance every obstacle one step (people walk, doors cycle)."""
        for obs in self.obstacles:
            obs.update(dt, rng=self._rng, bounds=self.ROOM_BOUNDS)

    def check_collision(self, position: np.ndarray):
        """Check a drone position against all obstacles.

        Returns (collided, obstacle_or_None).
        """
        for obs in self.obstacles:
            if obs.check_collision(position):
                return True, obs
        return False, None

    # -- lifecycle ------------------------------------------------------

    def reseed(self, rng: np.random.Generator) -> None:
        """Adopt the env's current RNG (env calls this on every reset)."""
        self._rng = rng
        self.randomizer.reseed(rng)

    def apply_randomization(self) -> Dict:
        """Sample episode params and push physics scales into dynamics."""
        self.dr_params = self.randomizer.apply_all()
        phys = self.dr_params["physics"]
        self.dynamics.params.mass = self._nominal_mass * phys.mass_scale
        self.dynamics.params.max_thrust = self._nominal_thrust * phys.thrust_scale
        self.dynamics.params.motor_time_constant = (
            self._nominal_tau * phys.motor_tau_scale
        )
        self.dynamics.gravity = np.array(
            [0.0, 0.0, -9.81 * phys.gravity_scale], dtype=np.float32
        )
        return self.dr_params

    def reset(self) -> DroneState:
        """Build the initial hover state (nominal physics, fresh pose)."""
        # Restore nominal physics first; env applies episode scales after.
        self.dynamics.params.mass = self._nominal_mass
        self.dynamics.params.max_thrust = self._nominal_thrust
        self.dynamics.params.motor_time_constant = self._nominal_tau
        self.dynamics.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        pos, quat = self.randomizer.randomize_initial_pose()
        hover = self.dynamics.compute_hover_thrust()
        state = DroneState(
            position=pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=quat,
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, hover, dtype=np.float32),
        )
        self._state_vec = state.to_array()
        self.randomizer.sample_wind(0.0)  # reset OU process
        # Sprint 5: spawn dynamic obstacles for this episode.
        self.spawn_obstacles()
        # Sprint 7: spawn victims for SAR mission.
        self._spawn_victims()
        return state

    def _spawn_victims(self) -> None:
        """Spawn victims for SAR mission."""
        self.victims = []
        self.medkit_carried = True
        self.victims_served = 0
        self.total_victims = 0
        if not self.sar_mission:
            return
        # Default: 4 victims at fixed positions in 4 rooms
        default_victims = [
            np.array([2.5, 2.5, 0.0], dtype=np.float32),
            np.array([5.5, 3.0, 0.0], dtype=np.float32),
            np.array([3.0, 7.0, 0.0], dtype=np.float32),
            np.array([7.0, 5.5, 0.0], dtype=np.float32),
        ]
        for v in default_victims:
            self.victims.append(Victim(v, rng=self._rng))
        self.total_victims = len(self.victims)

    def drop_payload(self, position: np.ndarray, radius_m: float = 2.0) -> bool:
        """SAR med-kit drop: serve the nearest unserved victim within radius.

        Returns True on a successful delivery (victim served, med-kit
        consumed), False when no med-kit is carried or no unserved victim
        is within `radius_m` (3D) of `position`. SAR-only, no weapons.
        """
        if not self.medkit_carried:
            return False
        pos = np.asarray(position, dtype=np.float32)
        for victim in self.victims:
            if victim.served:
                continue
            if float(np.linalg.norm(pos[:3] - np.asarray(victim.position, dtype=np.float32)[:3])) <= float(radius_m):
                victim.serve()
                self.medkit_carried = False
                self.victims_served += 1
                return True
        return False

    def step(
        self,
        action: np.ndarray,
        dt: float,
        goal: Optional[np.ndarray],
    ) -> Tuple[DroneState, float, bool, Dict]:
        """
        Integrate one control step. Returns (state, reward, done, info).
        Termination (crash/goal/bounds) is handled by BaseDroneEnv.step;
        this backend only reports physics-level done=False.
        """
        motor_cmd = np.clip(action, 0.0, 1.0).astype(np.float32)
        self._state_vec = self.dynamics.step(self._state_vec, motor_cmd, dt)

        # Wind drag: push position/velocity along the wind vector.
        wind = self.randomizer.sample_wind(dt)
        self._state_vec[0:3] += wind * dt * 0.3
        self._state_vec[3:6] += wind * dt * 0.1

        state = DroneState.from_array(self._state_vec)

        # Sprint 5: advance obstacles, then check drone-obstacle collision.
        self.step_obstacles(dt)
        collided, hit = self.check_collision(state.position)

        # Sprint 7: SAR mission - victim detection and med-kit drop
        victim_detected = False
        victim_served = False
        drop_medkit = False
        if self.sar_mission:
            # Check victim detection (within detection range)
            for victim in self.victims:
                if not victim.served:
                    dist_to_victim = float(np.linalg.norm(state.position[:2] - victim.position[:2]))
                    if dist_to_victim < 5.0:  # detection range
                        victim_detected = True
                        break

            # Check med-kit drop (action[4] would be drop signal, but we use proximity)
            # For mock: if drone is close to unserved victim and has medkit, auto-drop
            if self.medkit_carried:
                for victim in self.victims:
                    if not victim.served and victim.check_collision(state.position):
                        victim.serve()
                        self.medkit_carried = False
                        self.victims_served += 1
                        victim_served = True
                        drop_medkit = True
                        break

        # Curriculum-shaped reward
        cp = self._curriculum_params()
        hover = self.dynamics.compute_hover_thrust()
        effort = float(np.sum((motor_cmd - hover) ** 2))
        if goal is not None:
            dist = float(np.linalg.norm(state.position - goal))
            to_goal = goal - state.position
            dist_norm = max(dist, 1e-6)
            vel_toward = float(np.dot(state.velocity, to_goal) / dist_norm)
        else:
            dist = 0.0
            vel_toward = 0.0
        reward = cp["dist_w"] * dist + cp["vel_w"] * vel_toward + cp["effort_w"] * effort + cp["alive_w"]
        if collided:
            reward -= self.COLLISION_PENALTY
        # Sprint 7: SAR-specific rewards
        if self.sar_mission:
            if victim_detected:
                reward += 5.0  # bonus for detecting victim
            if victim_served:
                reward += 50.0  # large bonus for serving victim
            if drop_medkit:
                reward += 20.0  # bonus for dropping med-kit
            # Penalty for carrying medkit too long without serving
            if self.medkit_carried and self.victims_served < self.total_victims:
                reward -= 0.1

        info = {
            "backend": self.backend_name,
            "mock_backend": True,
            "wind": wind.copy(),
            "dist_to_goal": dist,
            "curriculum_level": self._curriculum_level,
            "goal_tolerance": cp["goal_tolerance"],
            "obstacle_collision": bool(collided),
            "num_obstacles": len(self.obstacles),
            # Sprint 7 SAR info
            "sar_mission": self.sar_mission,
            "victim_detected": victim_detected,
            "victim_served": victim_served,
            "victims_served": self.victims_served,
            "total_victims": self.total_victims,
            "medkit_carried": self.medkit_carried,
            "drop_medkit": drop_medkit,
        }
        return state, float(reward), False, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Synthesize a headless camera frame at config resolution."""
        res = tuple(getattr(self.config, "camera_resolution", (64, 64)))
        h, w = int(res[1]), int(res[0])
        alt = float(self._state_vec[2]) if self._state_vec is not None else 1.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0
        shade = np.clip(0.4 + 0.1 * alt + 0.2 * base, 0.0, 1.0)
        noise = self._rng.standard_normal((h, w)).astype(np.float32) * 0.03
        if mode == "depth_array":
            depth = np.clip(
                alt + (base - 0.5) * 2.0 + noise, 0.1, 100.0
            ).astype(np.float32)
            return depth
        # rgb_array default
        frame = np.stack([shade + noise, shade, shade - noise], axis=-1)
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)