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

if __name__ == '__main__':
    import argparse
    import zerorpc

    parser = argparse.ArgumentParser(description="Dummy MPPI bridge server")
    parser.add_argument('--address', default='tcp://0.0.0.0:4242',
                        help='zerorpc bind address')
    args = parser.parse_args()

    server = zerorpc.Server(DummyMPPIServer())
    server.bind(args.address)
    print(f"Dummy MPPI server listening on {args.address}")
    server.run()
