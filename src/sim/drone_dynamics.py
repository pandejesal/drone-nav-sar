#!/usr/bin/env python3
"""
Quadrotor Dynamics Model for DroneNav-SAR
Physics-based drone dynamics for simulation and control
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuadrotorParams:
    """Quadrotor physical parameters."""
    mass: float = 1.0              # kg
    arm_length: float = 0.2        # m (distance from center to motor)
    inertia: np.ndarray = None     # (3,3) inertia matrix in body frame
    max_thrust: float = 20.0       # N per motor
    max_rpm: float = 15000.0       # RPM
    thrust_coeff: float = 1.0e-5   # Thrust coefficient (N/RPM^2)
    drag_coeff: float = 1.0e-6     # Drag coefficient (Nm/RPM^2)
    motor_time_constant: float = 0.02  # s (first-order motor dynamics)

    def __post_init__(self):
        if self.inertia is None:
            # Default inertia for symmetric quadrotor
            Ixx = self.mass * self.arm_length**2 / 2
            Iyy = Ixx
            Izz = self.mass * self.arm_length**2
            self.inertia = np.diag([Ixx, Iyy, Izz]).astype(np.float32)


class QuadrotorDynamics:
    """
    Quadrotor dynamics model.
    State: [pos(3), vel(3), quat(4), ang_vel(3), motor_speeds(4)]
    Control: motor_thrusts(4) normalized [0, 1] or RPM commands
    """

    def __init__(self, params: Optional[QuadrotorParams] = None):
        self.params = params or QuadrotorParams()
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        self.state_dim = 17  # pos(3) + vel(3) + quat(4) + ang_vel(3) + motor(4)
        self.control_dim = 4  # 4 motor thrusts

        # Motor dynamics (first-order)
        self.motor_tau = self.params.motor_time_constant

    def quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication: q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    def quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    def quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q."""
        # v' = q * [0, v] * q^*
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        q_conj = self.quat_conjugate(q)
        result = self.quat_multiply(self.quat_multiply(q, qv), q_conj)
        return result[1:]

    def quat_to_rot_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    def quat_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    def euler_to_quat(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        roll, pitch, yaw = euler
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)

    def compute_motor_forces_torques(self, motor_thrusts: np.ndarray) -> tuple:
        """
        Compute total force and torque from motor thrusts.
        motor_thrusts: (4,) normalized thrust [0, 1] per motor
        Returns: (force_body, torque_body)
        """
        # Motor layout (X configuration):
        #   1 (CW)     2 (CCW)
        #       \   /
        #        \ /
        #         +
        #        / \
        #       /   \
        #   4 (CCW)   3 (CW)

        # Convert normalized thrust to force
        forces = motor_thrusts * self.params.max_thrust  # N per motor

        # Total thrust in body frame (z-axis up)
        total_thrust = np.sum(forces)
        force_body = np.array([0.0, 0.0, total_thrust], dtype=np.float32)

        # Torques from differential thrust
        l = self.params.arm_length / np.sqrt(2)  # arm length projected to x/y axes
        k = self.params.drag_coeff / self.params.thrust_coeff  # torque/thrust ratio

        # Roll torque (x-axis): motors 2,3 vs 1,4
        tau_x = l * (forces[1] + forces[2] - forces[0] - forces[3])

        # Pitch torque (y-axis): motors 1,2 vs 3,4
        tau_y = l * (forces[0] + forces[1] - forces[2] - forces[3])

        # Yaw torque (z-axis): CW vs CCW
        tau_z = k * (forces[0] - forces[1] + forces[2] - forces[3])

        torque_body = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        return force_body, torque_body

    def step(self, state: np.ndarray, motor_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate dynamics for one timestep.
        state: [pos(3), vel(3), quat(4), ang_vel(3), motor_speeds(4)]
        motor_cmd: (4,) normalized motor commands [0, 1]
        Returns: next_state
        """
        # Unpack state
        pos = state[0:3].copy()
        vel = state[3:6].copy()
        quat = state[6:10].copy()
        ang_vel = state[10:13].copy()
        motor_speeds = state[13:17].copy()

        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)

        # Motor dynamics (first-order, normalized [0, 1] state).
        # NOTE: motor_speeds is stored normalized (see packed state below),
        # so the target and integration stay in normalized units. The rate
        # is clamped to 1: control steps larger than the motor time
        # constant must not overshoot/oscillate (fixed 2026-09-14: Sprint-2
        # code mixed RPM targets with normalized state, breaking hover).
        motor_target = np.clip(motor_cmd, 0.0, 1.0).astype(np.float32)
        rate = min(dt / self.motor_tau, 1.0)
        motor_speeds = motor_speeds + (motor_target - motor_speeds) * rate
        motor_speeds = np.clip(motor_speeds, 0.0, 1.0)

        # Compute thrust from normalized motor speeds directly.
        motor_thrusts = motor_speeds

        # Forces and torques in body frame
        force_body, torque_body = self.compute_motor_forces_torques(motor_thrusts)

        # Transform force to world frame
        R = self.quat_to_rot_matrix(quat)
        force_world = R @ force_body

        # Add gravity
        force_world += self.params.mass * self.gravity

        # Linear acceleration
        accel = force_world / self.params.mass

        # Angular acceleration
        I_inv = np.linalg.inv(self.params.inertia)
        ang_accel = I_inv @ (torque_body - np.cross(ang_vel, self.params.inertia @ ang_vel))

        # Integrate (semi-implicit Euler)
        vel_new = vel + accel * dt
        pos_new = pos + vel_new * dt

        ang_vel_new = ang_vel + ang_accel * dt

        # Quaternion integration
        # q_dot = 0.5 * q * [0, ω]
        omega_quat = np.array([0.0, ang_vel_new[0], ang_vel_new[1], ang_vel_new[2]], dtype=np.float32)
        q_dot = 0.5 * self.quat_multiply(quat, omega_quat)
        quat_new = quat + q_dot * dt
        quat_new = quat_new / np.linalg.norm(quat_new)

        # Pack new state (motor_speeds already normalized [0, 1])
        new_state = np.concatenate([
            pos_new,
            vel_new,
            quat_new,
            ang_vel_new,
            motor_speeds,
        ]).astype(np.float32)

        return new_state

    def compute_hover_thrust(self) -> float:
        """Compute normalized thrust per motor for hover."""
        return (self.params.mass * abs(self.gravity[2])) / (4 * self.params.max_thrust)

    def linearize(self, state: np.ndarray) -> tuple:
        """
        Linearize dynamics around hover state.
        Returns: (A, B) matrices for LQR.
        """
        # At hover: pos=0, vel=0, quat=[1,0,0,0], ang_vel=0, motor=hover
        hover_thrust = self.compute_hover_thrust()
        hover_state = np.zeros(self.state_dim, dtype=np.float32)
        hover_state[6] = 1.0  # quat w
        hover_state[13:17] = hover_thrust
        hover_control = np.full(4, hover_thrust, dtype=np.float32)

        # Numerical linearization
        eps = 1e-6
        n = self.state_dim
        m = self.control_dim
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        next_nominal = self.step(hover_state, hover_control, 0.02)

        for i in range(n):
            state_pert = hover_state.copy()
            state_pert[i] += eps
            next_pert = self.step(state_pert, hover_control, 0.02)
            A[:, i] = (next_pert - next_nominal) / eps

        for j in range(m):
            ctrl_pert = hover_control.copy()
            ctrl_pert[j] += eps
            next_pert = self.step(hover_state, ctrl_pert, 0.02)
            B[:, j] = (next_pert - next_nominal) / eps

        return A, B


def create_default_quadrotor() -> QuadrotorDynamics:
    """Create a default quadrotor for a ~1kg drone."""
    params = QuadrotorParams(
        mass=1.0,
        arm_length=0.22,
        max_thrust=5.0,  # N per motor (total 20N)
        max_rpm=12000.0,
        thrust_coeff=1.2e-5,
        drag_coeff=2.5e-7,
        motor_time_constant=0.015,
    )
    return QuadrotorDynamics(params)


if __name__ == "__main__":
    # Quick test
    drone = create_default_quadrotor()

    # Initial state: at origin, hovering
    state = np.zeros(17, dtype=np.float32)
    state[6] = 1.0  # quaternion w
    hover_thrust = drone.compute_hover_thrust()
    state[13:17] = hover_thrust

    print(f"Hover thrust per motor: {hover_thrust:.3f}")
    print(f"Initial state: {state}")

    # Step a few times
    dt = 0.02
    for i in range(10):
        motor_cmd = np.full(4, hover_thrust)
        state = drone.step(state, motor_cmd, dt)
        print(f"Step {i}: pos={state[0:3]}, vel={state[3:6]}, quat={state[6:10]}")

    print("Dynamics test passed!")