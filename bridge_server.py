"""
Dummy MPPI bridge server for the puzzles repo.

Exposes the same zerorpc interface as MPPIIsaacLabPlanner so the real-robot
mppi_bridge_node can connect without modification.  All compute_action_tensor
calls return zero velocities; state received from the bridge is stored for
inspection but otherwise ignored.

Usage:
    python bridge_server.py [--address tcp://0.0.0.0:4242]
"""

import io
import json
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Transport helpers (mirrors mppi_bridge_node.py and isaaclab_mpc/utils/transport.py)
# ---------------------------------------------------------------------------

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b))


# ---------------------------------------------------------------------------
# Dummy server
# ---------------------------------------------------------------------------

NUM_JOINTS = 6


class DummyMPPIServer:
    """No-op MPPI planner server.

    Receives robot state, stores it, and returns zero velocity commands.
    """

    def __init__(self):
        self._goal: torch.Tensor = torch.zeros(3)
        self._latest_dof_state_bytes: bytes | None = None
        self._latest_object_states_bytes: bytes | None = None
        self._steps: list = []

    # --- Core planning -------------------------------------------------------

    def compute_action_tensor(
        self, dof_state_bytes: bytes, root_state_bytes: bytes
    ) -> bytes:
        self._latest_dof_state_bytes = dof_state_bytes

        dof_state = bytes_to_torch(dof_state_bytes)
        q  = dof_state[:NUM_JOINTS]
        dq = dof_state[NUM_JOINTS: NUM_JOINTS * 2]

        # Parse optional (pos[3], quat[4]) object blocks
        object_parts = []
        offset = NUM_JOINTS * 2
        while offset + 7 <= dof_state.numel():
            object_parts.append(dof_state[offset: offset + 7])
            offset += 7
        if object_parts:
            self._latest_object_states_bytes = torch_to_bytes(
                torch.cat(object_parts))

        # print(f"[dummy] q={q.tolist()} dq={dq.tolist()}")
        return torch_to_bytes(torch.zeros(1, NUM_JOINTS))

    # --- Goal ----------------------------------------------------------------

    def get_goal(self) -> bytes:
        return torch_to_bytes(self._goal)

    def set_goal(self, goal_bytes: bytes):
        self._goal = bytes_to_torch(goal_bytes).view(-1)[:3]
        print(f"[dummy] goal set to {self._goal.tolist()}")

    def get_current_goal_pos(self) -> bytes:
        if self._steps:
            return torch_to_bytes(
                torch.tensor(self._steps[0]["end_pos"], dtype=torch.float32))
        return torch_to_bytes(torch.zeros(3))

    # --- Episode / steps -----------------------------------------------------

    def reset_episode(self, steps_json: str = ""):
        self._steps = json.loads(steps_json) if steps_json else []
        print(f"[dummy] reset_episode: {len(self._steps)} step(s)")

    def get_current_step(self) -> bytes:
        return torch_to_bytes(torch.tensor(0.0))

    def get_total_steps(self) -> bytes:
        return torch_to_bytes(torch.tensor(float(len(self._steps))))

    # --- Robot / object state ------------------------------------------------

    def get_robot_state(self) -> bytes:
        if self._latest_dof_state_bytes is None:
            return torch_to_bytes(torch.zeros(NUM_JOINTS * 2))
        return self._latest_dof_state_bytes

    def set_object_states(self, obj_state_bytes: bytes):
        self._latest_object_states_bytes = obj_state_bytes

    def get_object_states(self) -> bytes:
        if self._latest_object_states_bytes is None:
            return torch_to_bytes(torch.zeros(0))
        return self._latest_object_states_bytes

    def get_sim_object_poses(self) -> bytes:
        return self.get_object_states()

    # --- MPPI metadata -------------------------------------------------------

    def get_rollouts(self) -> bytes:
        return torch_to_bytes(torch.zeros(1, 1, 3))

    def get_mppi_horizon(self) -> bytes:
        return torch_to_bytes(torch.tensor(1))

    def get_mppi_num_samples(self) -> bytes:
        return torch_to_bytes(torch.tensor(1))

    # --- Misc ----------------------------------------------------------------

    def update_weights(self, weights: dict):
        pass

    def test(self, msg: str):
        print(f"[dummy] test: {msg}")

    def get_scenario_info(self) -> str:
        return "{}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class FilteredDummyMPPIServer(DummyMPPIServer):
    """DummyMPPIServer with an EMA pose filter applied to incoming object states."""

    def __init__(self, alpha: float = 1.0, max_jump: float = 0.0):
        super().__init__()
        self._f_alpha    = alpha
        self._f_max_jump = max_jump
        self._f_states   = None  # list[(pos_tensor, quat_tensor)]
        if alpha < 1.0:
            print(f"[bridge_server] EMA filter: alpha={alpha:.2f}  max_jump={max_jump:.3f} m")

    def _filter(self, object_states):
        alpha    = self._f_alpha
        max_jump = self._f_max_jump
        if self._f_states is None:
            self._f_states = [(p.clone(), q.clone()) for p, q in object_states]
            return object_states
        # Resize state list if object count changed — add new entries, drop old ones.
        if len(self._f_states) != len(object_states):
            while len(self._f_states) < len(object_states):
                i = len(self._f_states)
                self._f_states.append((object_states[i][0].clone(), object_states[i][1].clone()))
            self._f_states = self._f_states[:len(object_states)]
        filtered = []
        for i, (pos, quat) in enumerate(object_states):
            prev_pos, prev_quat = self._f_states[i]
            if max_jump > 0.0 and (pos - prev_pos).norm().item() > max_jump:
                filtered.append((prev_pos, prev_quat))
                continue
            f_pos  = alpha * pos  + (1.0 - alpha) * prev_pos
            f_quat = alpha * quat + (1.0 - alpha) * prev_quat
            f_quat = f_quat / f_quat.norm().clamp(min=1e-8)
            self._f_states[i] = (f_pos, f_quat)
            filtered.append((f_pos, f_quat))
        return filtered

    def compute_action_tensor(self, dof_state_bytes: bytes, root_state_bytes: bytes) -> bytes:
        if self._f_alpha >= 1.0:
            return super().compute_action_tensor(dof_state_bytes, root_state_bytes)
        dof_state = bytes_to_torch(dof_state_bytes)
        offset = NUM_JOINTS * 2
        raw = []
        while offset + 7 <= dof_state.numel():
            raw.append((dof_state[offset: offset + 3], dof_state[offset + 3: offset + 7]))
            offset += 7
        if raw:
            filtered = self._filter(raw)
            parts = [dof_state[: NUM_JOINTS * 2]]
            for pos, quat in filtered:
                parts.append(pos)
                parts.append(quat)
            dof_state_bytes = torch_to_bytes(torch.cat(parts))
        return super().compute_action_tensor(dof_state_bytes, root_state_bytes)


class PoseEKF:
    """Kalman filter for a single object: constant-velocity model, position-only observations.

    State  x = [px, py, pz, vx, vy, vz]
    Process:  x_{k+1} = F(dt) x_k  +  w,   w ~ N(0, Q(dt, sigma_a))
    Observe:  z_k = H x_k          +  v,   v ~ N(0, R(sigma_r))

    Parameters
    ----------
    sigma_a : acceleration noise (m/s²) — how fast velocity can change.
              Low value → smooth but slow to follow pushes.
              High value → noisier but tracks rapid motion.
    sigma_r : position measurement noise (m, 1-sigma).
              Set to the expected std of your pose estimator.
    """

    _H = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0]], dtype=float)

    def __init__(self, sigma_a: float = 1.0, sigma_r: float = 0.02):
        self._sigma_a = sigma_a
        self._sigma_r = sigma_r
        self._R = np.eye(3) * sigma_r ** 2
        self._x: np.ndarray | None = None   # (6,)
        self._P: np.ndarray | None = None   # (6, 6)
        self._t: float | None = None

    def _FQ(self, dt: float):
        """State transition F and process noise Q for time step dt."""
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        sa2 = self._sigma_a ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = sa2 * np.array([
            [dt4 / 4, 0,       0,       dt3 / 2, 0,       0      ],
            [0,       dt4 / 4, 0,       0,       dt3 / 2, 0      ],
            [0,       0,       dt4 / 4, 0,       0,       dt3 / 2],
            [dt3 / 2, 0,       0,       dt2,     0,       0      ],
            [0,       dt3 / 2, 0,       0,       dt2,     0      ],
            [0,       0,       dt3 / 2, 0,       0,       dt2    ],
        ])
        return F, Q

    def update(self, pos: np.ndarray) -> np.ndarray:
        """Predict then update with a new position measurement. Returns filtered position (3,)."""
        now = time.monotonic()

        if self._x is None:
            self._x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0])
            self._P = np.diag([self._sigma_r ** 2] * 3 + [1.0] * 3)
            self._t = now
            return pos.copy()

        dt = max(now - self._t, 1e-3)
        self._t = now
        F, Q = self._FQ(dt)

        # Predict
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Update
        H = self._H
        innov = pos - H @ self._x
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(6) - K @ H) @ self._P

        return self._x[:3].copy()

    def reset(self):
        self._x = self._P = self._t = None


class OrientationEKF:
    """EKF for a single object's orientation.

    State  x = [qw, qx, qy, qz, ωx, ωy, ωz]
    Process:  constant angular velocity — q evolves via quaternion kinematics,
              ω is modelled as a random walk.
    Observe:  z = [qw, qx, qy, qz]  (direct quaternion measurement)

    The nonlinearity is in the process model (F depends on ω), so this is a
    proper EKF with a linearised Jacobian at each predict step.

    Parameters
    ----------
    sigma_w   : angular velocity process noise (rad/s) — how fast ω can change.
    sigma_rot : quaternion measurement noise (dimensionless) — roughly sin(σ_angle/2).
                ~0.01 ≈ ±1°, ~0.05 ≈ ±5°, ~0.1 ≈ ±11°.
    """

    def __init__(self, sigma_w: float = 0.5, sigma_rot: float = 0.05):
        self._sigma_w   = sigma_w
        self._sigma_rot = sigma_rot
        self._R = np.eye(4) * sigma_rot ** 2
        # H: observe quaternion directly, not angular velocity
        self._H = np.zeros((4, 7))
        self._H[:4, :4] = np.eye(4)
        self._x: np.ndarray | None = None   # (7,)
        self._P: np.ndarray | None = None   # (7, 7)
        self._t: float | None      = None

    @staticmethod
    def _omega_mat(w: np.ndarray) -> np.ndarray:
        """4×4 Omega(ω): qdot = Omega(ω) @ q  (pure-quaternion kinematics, pre-factor 0.5 omitted here)."""
        wx, wy, wz = w
        return np.array([
            [ 0,  -wx, -wy, -wz],
            [wx,    0,  wz, -wy],
            [wy,  -wz,   0,  wx],
            [wz,   wy, -wx,   0],
        ])

    @staticmethod
    def _G_mat(q: np.ndarray) -> np.ndarray:
        """4×3 G(q): ∂(qdot)/∂ω = 0.5 * G(q)  so ∂q_new/∂ω = 0.5*dt*G(q)."""
        qw, qx, qy, qz = q
        return np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
        ])

    def _FQ(self, dt: float):
        q = self._x[:4]
        w = self._x[4:]
        Ow = self._omega_mat(w)
        G  = self._G_mat(q)

        # Linearised process Jacobian (7×7)
        F = np.eye(7)
        F[:4, :4] = np.eye(4) + 0.5 * dt * Ow   # ∂q_new/∂q
        F[:4, 4:] = 0.5 * dt * G                  # ∂q_new/∂ω
        # ∂ω/∂ω = I (already set)

        # Process noise
        sw2 = self._sigma_w ** 2
        Q = np.zeros((7, 7))
        Q[:4, :4] = sw2 * (0.5 * dt) ** 2 * G @ G.T   # orientation uncertainty from ω noise
        Q[4:, 4:] = sw2 * np.eye(3)                     # angular velocity random walk

        return F, Q

    def update(self, quat: np.ndarray) -> np.ndarray:
        """Predict then update with a raw quaternion measurement. Returns filtered quaternion (4,)."""
        now = time.monotonic()
        quat = np.asarray(quat, dtype=float)
        quat /= np.linalg.norm(quat)

        if self._x is None:
            self._x = np.concatenate([quat, [0.0, 0.0, 0.0]])
            self._P = np.diag([self._sigma_rot ** 2] * 4 + [1.0] * 3)
            self._t = now
            return quat.copy()

        # Enforce quaternion sign consistency (q and -q are the same rotation)
        if np.dot(quat, self._x[:4]) < 0:
            quat = -quat

        dt = max(now - self._t, 1e-3)
        self._t = now
        F, Q = self._FQ(dt)

        # Predict
        q, w = self._x[:4], self._x[4:]
        q_pred = q + 0.5 * dt * (self._omega_mat(w) @ q)
        q_pred /= np.linalg.norm(q_pred)
        self._x[:4] = q_pred
        self._P = F @ self._P @ F.T + Q

        # Update
        H = self._H
        innov = quat - H @ self._x
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x += K @ innov
        self._x[:4] /= np.linalg.norm(self._x[:4])   # re-normalise
        self._P = (np.eye(7) - K @ H) @ self._P

        return self._x[:4].copy()

    def reset(self):
        self._x = self._P = self._t = None


class EKFDummyMPPIServer(DummyMPPIServer):
    """DummyMPPIServer with a per-object Kalman filter on position.

    Quaternion is smoothed separately with EMA (quat_alpha).

    Parameters
    ----------
    sigma_a    : KF process noise — acceleration uncertainty (m/s²).
    sigma_r    : KF measurement noise — pose estimator std (m).
    quat_alpha : EMA weight for quaternion (1.0 = raw, 0.0 = frozen).
    """

    def __init__(self, sigma_a: float = 1.0, sigma_r: float = 0.02,
                 quat_alpha: float = 0.3):
        super().__init__()
        self._sigma_a    = sigma_a
        self._sigma_r    = sigma_r
        self._quat_alpha = quat_alpha
        self._ekfs:        list[PoseEKF]           = []
        self._prev_quats:  list[torch.Tensor | None] = []
        print(f"[bridge_server] EKF filter: sigma_a={sigma_a} m/s²  "
              f"sigma_r={sigma_r} m  quat_alpha={quat_alpha}")

    def _ensure_ekfs(self, n: int):
        while len(self._ekfs) < n:
            self._ekfs.append(PoseEKF(self._sigma_a, self._sigma_r))
            self._prev_quats.append(None)

    def compute_action_tensor(self, dof_state_bytes: bytes, root_state_bytes: bytes) -> bytes:
        dof_state = bytes_to_torch(dof_state_bytes)
        offset = NUM_JOINTS * 2
        raw = []
        while offset + 7 <= dof_state.numel():
            raw.append((dof_state[offset: offset + 3], dof_state[offset + 3: offset + 7]))
            offset += 7

        if raw:
            self._ensure_ekfs(len(raw))
            parts = [dof_state[:NUM_JOINTS * 2]]
            for i, (pos, quat) in enumerate(raw):
                # Position: Kalman filter
                filtered_pos = self._ekfs[i].update(pos.numpy().astype(float))
                f_pos = torch.tensor(filtered_pos, dtype=torch.float32)

                # Quaternion: EMA
                a = self._quat_alpha
                if self._prev_quats[i] is None:
                    f_quat = quat.clone()
                else:
                    f_quat = a * quat + (1.0 - a) * self._prev_quats[i]
                    f_quat = f_quat / f_quat.norm().clamp(min=1e-8)
                self._prev_quats[i] = f_quat.detach()

                parts.append(f_pos)
                parts.append(f_quat)
            dof_state_bytes = torch_to_bytes(torch.cat(parts))

        return super().compute_action_tensor(dof_state_bytes, root_state_bytes)


class FullEKFDummyMPPIServer(DummyMPPIServer):
    """DummyMPPIServer with independent EKF filters for both position and orientation.

    Position  : constant-velocity EKF via PoseEKF.
    Orientation: constant-angular-velocity EKF via OrientationEKF.

    Parameters
    ----------
    sigma_a   : position KF — acceleration process noise (m/s²).
    sigma_r   : position KF — measurement noise (m).
    sigma_w   : orientation KF — angular velocity process noise (rad/s).
    sigma_rot : orientation KF — quaternion measurement noise.
    """

    def __init__(self, sigma_a: float = 1.0, sigma_r: float = 0.02,
                 sigma_w: float = 0.5, sigma_rot: float = 0.05):
        super().__init__()
        self._sigma_a   = sigma_a
        self._sigma_r   = sigma_r
        self._sigma_w   = sigma_w
        self._sigma_rot = sigma_rot
        self._pos_ekfs:  list[PoseEKF]        = []
        self._ori_ekfs:  list[OrientationEKF] = []
        print(f"[bridge_server] Full-EKF filter: "
              f"sigma_a={sigma_a} m/s²  sigma_r={sigma_r} m  "
              f"sigma_w={sigma_w} rad/s  sigma_rot={sigma_rot}")

    def _ensure_ekfs(self, n: int):
        while len(self._pos_ekfs) < n:
            self._pos_ekfs.append(PoseEKF(self._sigma_a, self._sigma_r))
            self._ori_ekfs.append(OrientationEKF(self._sigma_w, self._sigma_rot))

    def compute_action_tensor(self, dof_state_bytes: bytes, root_state_bytes: bytes) -> bytes:
        dof_state = bytes_to_torch(dof_state_bytes)
        offset = NUM_JOINTS * 2
        raw = []
        while offset + 7 <= dof_state.numel():
            raw.append((dof_state[offset: offset + 3], dof_state[offset + 3: offset + 7]))
            offset += 7

        if raw:
            self._ensure_ekfs(len(raw))
            parts = [dof_state[:NUM_JOINTS * 2]]
            for i, (pos, quat) in enumerate(raw):
                filtered_pos  = self._pos_ekfs[i].update(pos.numpy().astype(float))
                filtered_quat = self._ori_ekfs[i].update(quat.numpy().astype(float))
                parts.append(torch.tensor(filtered_pos,  dtype=torch.float32))
                parts.append(torch.tensor(filtered_quat, dtype=torch.float32))
            dof_state_bytes = torch_to_bytes(torch.cat(parts))

        return super().compute_action_tensor(dof_state_bytes, root_state_bytes)


if __name__ == '__main__':
    import argparse
    import zerorpc

    parser = argparse.ArgumentParser(description="Dummy MPPI bridge server")
    parser.add_argument('--address',    default='tcp://0.0.0.0:4242')
    parser.add_argument('--filter',     choices=['none', 'ema', 'ekf', 'ekf_full'], default='none',
                        help='Pose filter type')
    # EMA options
    parser.add_argument('--alpha',      type=float, default=0.1,
                        help='EMA weight for new poses (1.0 = off)')
    parser.add_argument('--max_jump',   type=float, default=0.0,
                        help='EMA outlier gate in metres (0.0 = off)')
    # EKF position options
    parser.add_argument('--sigma_a',    type=float, default=1.0,
                        help='EKF: acceleration process noise (m/s²)')
    parser.add_argument('--sigma_r',    type=float, default=0.02,
                        help='EKF: position measurement noise (m)')
    parser.add_argument('--quat_alpha', type=float, default=0.3,
                        help='EKF (position-only): EMA weight for quaternion (1.0 = raw)')
    # EKF orientation options (ekf_full only)
    parser.add_argument('--sigma_w',    type=float, default=0.5,
                        help='EKF_full: angular velocity process noise (rad/s)')
    parser.add_argument('--sigma_rot',  type=float, default=0.05,
                        help='EKF_full: quaternion measurement noise')
    args = parser.parse_args()

    if args.filter == 'ekf_full':
        srv = FullEKFDummyMPPIServer(sigma_a=args.sigma_a, sigma_r=args.sigma_r,
                                     sigma_w=args.sigma_w, sigma_rot=args.sigma_rot)
    elif args.filter == 'ekf':
        srv = EKFDummyMPPIServer(sigma_a=args.sigma_a, sigma_r=args.sigma_r,
                                 quat_alpha=args.quat_alpha)
    elif args.filter == 'ema':
        srv = FilteredDummyMPPIServer(alpha=args.alpha, max_jump=args.max_jump)
    else:
        srv = DummyMPPIServer()

    server = zerorpc.Server(srv)
    server.bind(args.address)
    print(f"Dummy MPPI server listening on {args.address}  filter={args.filter}")
    server.run()
