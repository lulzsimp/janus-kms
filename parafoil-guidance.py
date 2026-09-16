"""
Parafoil 6-DOF simulation with proportional heading guidance.

The model is a compact engineering simulation based on the supplied
paper-derived equations. It contains:

* a 12-state nonlinear 6-DOF plant,
* body/inertial coordinate transforms,
* aerodynamic force and moment estimates,
* fourth-order Runge-Kutta integration,
* proportional heading guidance,
* command saturation and actuator-rate limiting,
* terminal behavior near the target.

Coordinate convention:
    inertial x = north, y = east, z = down
    therefore z = -700 m means 700 m above the landing plane z = 0.

The inertia values and several geometry values are explicit engineering
assumptions because the supplied paper/code does not provide a complete
calibrated parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

RHO = 1.225
GRAVITY = 9.81
MASS = 4.5

WING_AREA = 3.0
SPAN = 3.0
CHORD = 1.0
CONTROL_ARM = 0.1

# Aerodynamic coefficients from the supplied tables.
CL0 = 0.5
CL_ALPHA = 1.719
CL_DELTA = 0.0001

CD0 = 0.2
CD_ALPHA2 = 0.7
CD_DELTA = 0.0001

CM0 = 0.1397
CM_ALPHA = -1.4308
CM_Q = -0.2251

CL_PHI = -0.04
CL_P = -0.08
CL_DELTA_A = -0.00001

CN_R = -0.012
CN_DELTA_A = -0.00008

# Engineering assumptions; replace with measured/calibrated values.
INERTIA = np.array(
    [
        [1.0, 0.0, 0.05],
        [0.0, 2.0, 0.0],
        [0.05, 0.0, 1.5],
    ],
    dtype=float,
)

INERTIA_INV = np.linalg.inv(INERTIA)

# Guidance and numerical parameters.
MAX_BRAKE = 0.20
KP_HEADING = 0.80
HEADING_DEADBAND = np.deg2rad(2.0)
GOAL_RADIUS = 10.0
TERMINAL_DECAY = 0.90
ACTUATOR_RATE = 0.40       # command units per second

GUIDANCE_DT = 0.50
PHYSICS_DT = 0.02
MAX_TIME = 500.0

# Wind expressed in inertial coordinates [north, east, down].
WIND_INERTIAL = np.zeros(3)


@dataclass
class GuidanceResult:
    command: float
    mode: str


def body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return the rotation matrix mapping body vectors to inertial vectors."""
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    ss, cs = np.sin(psi), np.cos(psi)

    inertial_to_body = np.array(
        [
            [ct * cs, ct * ss, -st],
            [
                sp * st * cs - cp * ss,
                sp * st * ss + cp * cs,
                sp * ct,
            ],
            [
                cp * st * cs + sp * ss,
                cp * st * ss - sp * cs,
                cp * ct,
            ],
        ]
    )

    return inertial_to_body.T


def euler_rates(
    phi: float,
    theta: float,
    p: float,
    q: float,
    r: float,
) -> np.ndarray:
    """Convert body angular rates into roll, pitch, and yaw rates."""
    sp, cp = np.sin(phi), np.cos(phi)

    cos_theta = np.cos(theta)
    cos_theta = np.copysign(max(abs(cos_theta), 1e-6), cos_theta)
    tan_theta = np.sin(theta) / cos_theta

    return np.array(
        [
            p + sp * tan_theta * q + cp * tan_theta * r,
            cp * q - sp * r,
            sp * q / cos_theta + cp * r / cos_theta,
        ]
    )


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dynamics(state: np.ndarray, command: float) -> np.ndarray:
    """
    Compute the derivative of the 12-element state.

    State order:
        [x, y, z, phi, theta, psi, u, v, w, p, q, r]

    The signed command controls roll/yaw steering moments. Its magnitude is
    used for the common lift/drag control contribution.
    """
    (
        _x,
        _y,
        _z,
        phi,
        theta,
        psi,
        u,
        v,
        w,
        p,
        q,
        r,
    ) = state

    velocity_body = np.array([u, v, w], dtype=float)
    angular_rate = np.array([p, q, r], dtype=float)

    body_to_world = body_to_inertial(phi, theta, psi)
    world_to_body = body_to_world.T

    velocity_world = body_to_world @ velocity_body

    air_velocity_world = velocity_world - WIND_INERTIAL
    air_velocity_body = world_to_body @ air_velocity_world

    ua, va, wa = air_velocity_body

    airspeed = max(np.linalg.norm(air_velocity_body), 1e-8)
    alpha = np.arctan2(wa, max(ua, 1e-8))

    command_magnitude = abs(command)

    lift_coefficient = (
        CL0
        + CL_ALPHA * alpha
        + CL_DELTA * command_magnitude
    )

    drag_coefficient = (
        CD0
        + CD_ALPHA2 * alpha**2
        + CD_DELTA * command_magnitude
    )

    dynamic_pressure_area = (
        0.5 * RHO * WING_AREA * airspeed**2
    )

    lift = (
        dynamic_pressure_area
        * lift_coefficient
        * np.array([wa, 0.0, -ua])
    )

    drag = (
        dynamic_pressure_area
        * drag_coefficient
        * np.array([ua, va, wa])
    )

    aerodynamic_force = lift - drag

    gravity_body = world_to_body @ np.array(
        [0.0, 0.0, MASS * GRAVITY]
    )

    aerodynamic_moment = dynamic_pressure_area * np.array(
        [
            (
                CL_PHI * phi
                + CL_P * SPAN**2 * p / (2.0 * airspeed)
                + CL_DELTA_A
                * command
                * SPAN
                / CONTROL_ARM
            ),
            (
                CM0 * CHORD
                + CM_ALPHA * CHORD * alpha
                + CM_Q * CHORD**2 * q / (2.0 * airspeed)
            ),
            (
                CN_R * SPAN**2 * r / (2.0 * airspeed)
                + CN_DELTA_A
                * command
                * SPAN
                / CONTROL_ARM
            ),
        ]
    )

    velocity_body_dot = (
        (gravity_body + aerodynamic_force) / MASS
        - np.cross(angular_rate, velocity_body)
    )

    angular_rate_dot = INERTIA_INV @ (
        aerodynamic_moment
        - np.cross(
            angular_rate,
            INERTIA @ angular_rate,
        )
    )

    return np.concatenate(
        [
            velocity_world + WIND_INERTIAL,
            euler_rates(phi, theta, p, q, r),
            velocity_body_dot,
            angular_rate_dot,
        ]
    )


def apply_rate_limit(
    previous: float,
    desired: float,
    dt: float,
) -> float:
    """Prevent the brake command from changing instantaneously."""
    maximum_change = ACTUATOR_RATE * dt

    change = np.clip(
        desired - previous,
        -maximum_change,
        maximum_change,
    )

    return float(
        np.clip(
            previous + change,
            -MAX_BRAKE,
            MAX_BRAKE,
        )
    )


def guidance(
    state: np.ndarray,
    previous_command: float,
) -> GuidanceResult:
    """
    Compute a bounded proportional steering command.

    The controller points the parafoil toward the target by comparing the
    desired target bearing against the current horizontal velocity track.
    """
    x, y = state[0], state[1]
    phi, theta, psi = state[3:6]
    body_velocity = state[6:9]

    distance = float(np.hypot(x, y))

    if distance <= GOAL_RADIUS:
        return GuidanceResult(
            command=TERMINAL_DECAY * previous_command,
            mode="TERMINAL",
        )

    body_to_world = body_to_inertial(phi, theta, psi)
    velocity_world = body_to_world @ body_velocity

    horizontal_velocity = velocity_world[:2]
    horizontal_speed = np.linalg.norm(horizontal_velocity)

    if horizontal_speed < 0.2:
        current_track = psi
    else:
        current_track = np.arctan2(
            horizontal_velocity[1],
            horizontal_velocity[0],
        )

    desired_bearing = np.arctan2(-y, -x)

    heading_error = wrap_pi(
        desired_bearing - current_track
    )

    if abs(heading_error) <= HEADING_DEADBAND:
        desired_command = 0.0
    else:
        # The negative sign matches the assumed moment/heading convention.
        desired_command = -KP_HEADING * heading_error

    desired_command = float(
        np.clip(
            desired_command,
            -MAX_BRAKE,
            MAX_BRAKE,
        )
    )

    command = apply_rate_limit(
        previous_command,
        desired_command,
        GUIDANCE_DT,
    )

    return GuidanceResult(
        command=command,
        mode="CRUISE",
    )


def rk4_step(
    state: np.ndarray,
    command: float,
    dt: float,
) -> np.ndarray:
    """Advance the state by one fourth-order Runge-Kutta step."""
    k1 = dynamics(state, command)

    k2 = dynamics(
        state + 0.5 * dt * k1,
        command,
    )

    k3 = dynamics(
        state + 0.5 * dt * k2,
        command,
    )

    k4 = dynamics(
        state + dt * k3,
        command,
    )

    return state + (
        dt
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
        / 6.0
    )


def initial_state(seed: int = 7) -> np.ndarray:
    """Create a randomized launch point using the paper's initial motion."""
    rng = np.random.default_rng(seed)
    x0, y0 = rng.uniform(-250.0, 250.0, size=2)

    return np.array(
        [
            x0,
            y0,
            -700.0,
            0.0,
            0.0,
            0.0,
            6.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def simulate(
    seed: int = 7,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run the simulation until landing or the time limit."""
    state = initial_state(seed)

    time = 0.0
    command = 0.0
    next_guidance_update = 0.0
    mode = "CRUISE"

    times = [time]
    states = [state.copy()]
    commands = [command]
    modes = [mode]

    while time < MAX_TIME and state[2] < 0.0:
        if time >= next_guidance_update - 1e-12:
            result = guidance(state, command)
            command = result.command
            mode = result.mode
            next_guidance_update += GUIDANCE_DT

        dt = min(
            PHYSICS_DT,
            MAX_TIME - time,
            next_guidance_update - time,
        )

        if dt <= 1e-12:
            continue

        state = rk4_step(state, command, dt)
        time += dt

        times.append(time)
        states.append(state.copy())
        commands.append(command)
        modes.append(mode)

    return (
        np.asarray(times),
        np.asarray(states),
        np.asarray(commands),
        np.asarray(modes),
    )


def main() -> None:
    """Run one example and display trajectory/controller diagnostics."""
    times, states, commands, modes = simulate(seed=7)

    landing_point = states[-1, :2]
    miss_distance = float(np.linalg.norm(landing_point))

    print(
        f"Launch point: "
        f"{states[0, 0]:.2f}, "
        f"{states[0, 1]:.2f}, "
        f"{states[0, 2]:.2f} m"
    )

    print(
        f"Final point:  "
        f"{states[-1, 0]:.2f}, "
        f"{states[-1, 1]:.2f}, "
        f"{states[-1, 2]:.2f} m"
    )

    print(
        f"Horizontal miss distance: "
        f"{miss_distance:.2f} m"
    )

    print(f"Flight time: {times[-1]:.2f} s")
    print(f"Final guidance mode: {modes[-1]}")

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot(
        states[:, 0],
        states[:, 1],
        states[:, 2],
        label="Parafoil trajectory",
    )

    axis.scatter(
        [states[0, 0]],
        [states[0, 1]],
        [states[0, 2]],
        label="Launch",
    )

    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=90,
        label="Target",
    )

    axis.set_xlabel("North x [m]")
    axis.set_ylabel("East y [m]")
    axis.set_zlabel("Down z [m]")
    axis.set_title("6-DOF Parafoil Guidance")
    axis.legend()

    figure.tight_layout()

    plt.figure(figsize=(9, 4))
    plt.plot(times, commands)
    plt.xlabel("Time [s]")
    plt.ylabel("Signed brake command")
    plt.title("Proportional Heading Guidance")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
