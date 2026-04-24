"""
IsaacLab implementation of the bin environment.

Mirrors genesis_env.py: same bin layout, action primitives, reward/goal logic —
only the simulator API differs.

Bin layout (top-down, z-up):
  - Floor at z=0
  - North wall: +y side
  - West wall:  -x side
  - East wall:  +x side
  - OPEN south side: -y exit
  - Objects start inside the bin

Supports single-env mode (n_envs=1) and parallel mode (n_envs>1).

For parallel envs, each env is offset in world space on a grid; all state
dicts returned to the user are in bin-local frame (origin at bin SW corner)
so they are interchangeable with genesis_env.py state dicts.

Note on Isaac Lab startup: Isaac Lab requires the Omniverse / Isaac Sim
application to be running.  Launch it before instantiating this class with:

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)   # or headless=False for viewer
    simulation_app = launcher.app

Quaternion convention: (w, x, y, z) — matches Genesis.
"""

import numpy as np
import torch
from typing import List, Optional
from .base_env import SimulatorEnv

try:
    import os as _os
    from isaaclab.app import AppLauncher
    _headless        = _os.environ.get('ISAACLAB_HEADLESS',        '0') == '1'
    _enable_cameras  = _os.environ.get('ISAACLAB_ENABLE_CAMERAS',  '0') == '1'
    app_launcher = AppLauncher(headless=_headless, enable_cameras=_enable_cameras)
    simulation_app = app_launcher.app


    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        # Older package name (Isaac Lab < 2.0)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim import SimulationContext, SimulationCfg
        from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
        from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
        ISAACLAB_AVAILABLE = True
    except ImportError:
        ISAACLAB_AVAILABLE = False
        sim_utils = None
        SimulationContext = None
        SimulationCfg = None
        RigidObject = None
        RigidObjectCfg = None
        ContactSensor = None
        ContactSensorCfg = None

# Bin dimensions (identical to genesis_env.py)
BIN_W  = 1.0    # x extent
BIN_D  = 1.0    # y extent (depth, from 0 to BIN_D)
BIN_H  = 0.5    # wall height
WALL_T = 0.05   # wall thickness

OBJ_SIZE = 0.08
OBJ_H    = OBJ_SIZE / 2   # object centre z when resting on floor

PUSHER_T = 0.012
PUSHER_W = OBJ_SIZE * 0.88

EXIT_Y = -0.05   # target exits when its y < EXIT_Y
_PARK  = [BIN_W / 2, -2.0, OBJ_H]   # pusher parking spot (bin-local)

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)   # (w, x, y, z)


class BinEnvIsaacLab(SimulatorEnv):
    """
    IsaacLab implementation of the bin environment with pushers and objects.

    Two thin pusher blades are kinematically controlled each physics step by
    writing their root-pose directly via RigidObject.write_root_pose_to_sim().

    For parallel envs, each env is placed in its own region of world space
    (grid layout); returned state positions are always in bin-local frame.

    Parameters
    ----------
    n_obstacles : int
    n_envs : int
        1 = single mode (checkpoint reset, viewer support).
        >1 = parallel mode (GPU batch evaluation).
    show_viewer : bool
        Only used when n_envs=1 and the app was launched with headless=False.
    dt : float
    seed : int | None
    stackable : bool
    friction : float
    n_z_levels : int
    push_steps : int
    substeps : int
    """

    def __init__(self, n_obstacles: int = 2, n_envs: int = 1,
                 show_viewer: bool = False,
                 dt: float = 0.01, seed: int | None = None,
                 stackable: bool = False, friction: float = 1.0,
                 n_z_levels: int = 1,
                 push_steps: int = 20, substeps: int = 4,
                 wall_thickness: float = WALL_T,
                 difficult_spawn: bool = False,
                 reward_cfg=None,
                 bin_size: float | None = None,
                 bin_size_factor: float = 0.9,
                 force_threshold: float = 100.0,
                 debug: bool = False,
                 target_z_level: int | None = None):
        if not ISAACLAB_AVAILABLE:
            raise ImportError(
                "isaaclab (or omni.isaac.lab) is not installed. "
                "Install Isaac Lab before using BinEnvIsaacLab."
            )

        super().__init__(
            n_obstacles=n_obstacles, n_envs=n_envs,
            show_viewer=show_viewer, dt=dt, seed=seed,
            stackable=stackable, friction=friction,
            n_z_levels=n_z_levels, push_steps=push_steps,
            substeps=substeps, wall_thickness=wall_thickness,
            difficult_spawn=difficult_spawn, reward_cfg=reward_cfg,
            bin_size=bin_size, bin_size_factor=bin_size_factor,
            debug=debug, target_z_level=target_z_level,
        )

        self.force_threshold = force_threshold
        self.force_trace: list[float] = []  # per-step force magnitudes for env 0, last batch_evaluate
        self._post_step_hook = None   # callable invoked after every _step_sim; used by record_replay
        self._force_render   = False  # when True, _step_sim always renders (used by replay)

        _park_y = -(max(self.bin_w, self.bin_d) * 1.5 + 0.1)
        self._park = [self.bin_w / 2, _park_y, OBJ_H]
        self.z_levels = [OBJ_H + i * OBJ_SIZE for i in range(n_z_levels)]
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"

        # Compute per-env world origins so envs don't overlap.
        # Account for wall thickness so thick walls don't cause env overlap.
        env_spacing = (max(self.bin_w, self.bin_d) + 2 * self.wall_thickness) * 2.5
        n_cols      = max(1, int(np.ceil(np.sqrt(self.n_envs))))
        self.env_origins = torch.zeros(self.n_envs, 3, device=self.device)
        for ei in range(self.n_envs):
            row = ei // n_cols
            col = ei % n_cols
            self.env_origins[ei, 0] = col * env_spacing
            self.env_origins[ei, 1] = row * env_spacing

        # Checkpoint for single-mode reset (saved after _place_objects settles)
        self._initial_states: dict | None = None

        # Rendering toggle: press 'f' in the viewer to flip this flag
        self.rendering_enabled: bool = True

        self._init_sim()
        self._build_scene()
        # Initialise the PhysX backend; must precede any tensor reads/writes
        self.sim.reset()
        self._refresh_all()

        # Always place objects in env 0 — this produces the initial_state that
        # main.py samples via get_state(0) before handing it to the planner.
        # In parallel mode the other env slots are populated by batch_evaluate
        # via _set_state, so they don't need independent placement here.
        self._place_objects()

    # ------------------------------------------------------------------
    # Simulator initialisation
    # ------------------------------------------------------------------

    def _init_sim(self):
        """Create the SimulationContext (wraps Isaac Sim's PhysX stage)."""
        sim_cfg = SimulationCfg(
            dt=self.dt,
            gravity=(0.0, 0.0, -9.81),
            device=self.device,
            physx=PhysxCfg(
                enable_ccd=False,
                min_position_iteration_count=16,
                min_velocity_iteration_count=4,
                enable_external_forces_every_iteration=True,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        if self.show_viewer:
            ox = self.env_origins[0, 0].item()
            oy = self.env_origins[0, 1].item()
            eye, target = self._camera_view(ox, oy)
            # set_camera_view is available in Isaac Lab >= 1.0
            self.sim.set_camera_view(
                eye=np.array(eye),
                target=np.array(target),
            )
            self._start_render_toggle_thread()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _make_box_cfg(self, size: tuple, mass: float, kinematic: bool,
                      friction: float, color: tuple | None = None,
                      pos_iters: int | None = None,
                      contact_offset: float = 0.005,
                      activate_contact_sensors: bool = False):
        """Return a CuboidCfg for spawning a rigid box prim."""
        vis = (sim_utils.PreviewSurfaceCfg(diffuse_color=color)
               if color is not None else None)
        return sim_utils.CuboidCfg(
            size=size,
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                disable_gravity=kinematic,
                solver_position_iteration_count=pos_iters,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=contact_offset,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            ),
            visual_material=vis,
        )

    def _spawn_prim(self, prim_path: str, cfg,
                    pos: tuple, quat: tuple = _IDENTITY_QUAT):
        """Spawn a prim at the given world position and orientation (w,x,y,z)."""
        cfg.func(prim_path, cfg, translation=pos, orientation=quat)

    def _build_scene(self):
        """
        Spawn all geometry prims, then wrap dynamic objects in RigidObject views.

        Each env gets its own set of prims under /World/envs/env_{i}/ so that
        collision groups are naturally separated per env.
        Static geometry (floor, walls) is spawned as kinematic rigid bodies.
        Dynamic objects (pushers, target, obstacles) are spawned and then
        wrapped with pattern-path RigidObject views for batched tensor access.
        """
        fr = self.friction
        wt = self.wall_thickness

        bw, bd = self.bin_w, self.bin_d

        # Config templates (size is x, y, z)
        floor_cfg  = self._make_box_cfg(
            (bw + 2*wt, bd + 2*wt, wt),
            mass=1.0, kinematic=True, friction=fr, color=(0.7, 0.6, 0.5))
        north_cfg  = self._make_box_cfg(
            (bw + 2*wt, wt, BIN_H),
            mass=1.0, kinematic=True, friction=fr, color=(0.5, 0.5, 0.8),
            contact_offset=0.02)
        west_cfg   = self._make_box_cfg(
            (wt, bd, BIN_H),
            mass=1.0, kinematic=True, friction=fr, color=(0.5, 0.5, 0.8),
            contact_offset=0.02)
        east_cfg   = self._make_box_cfg(
            (wt, bd, BIN_H),
            mass=1.0, kinematic=True, friction=fr, color=(0.5, 0.5, 0.8),
            contact_offset=0.02)
        pns_cfg    = self._make_box_cfg(
            (PUSHER_W, PUSHER_T, PUSHER_W),
            mass=10.0, kinematic=True, friction=fr, color=(0.2, 0.9, 0.2),
            activate_contact_sensors=True)
        pew_cfg    = self._make_box_cfg(
            (PUSHER_T, PUSHER_W, PUSHER_W),
            mass=10.0, kinematic=True, friction=fr, color=(0.9, 0.6, 0.1),
            activate_contact_sensors=True)
        target_cfg = self._make_box_cfg(
            (OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
            mass=0.05, kinematic=False, friction=fr, color=(0.9, 0.2, 0.2),
            pos_iters=8)
        obs_cfg    = self._make_box_cfg(
            (OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
            mass=0.5, kinematic=False, friction=fr, color=(0.3, 0.5, 0.9),
            pos_iters=8)

        # Ground plane (one shared plane under all envs)
        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

        for ei in range(self.n_envs):
            ox = self.env_origins[ei, 0].item()
            oy = self.env_origins[ei, 1].item()
            ep = f"/World/envs/env_{ei}"

            # Static geometry
            self._spawn_prim(f"{ep}/Floor", floor_cfg,
                             (ox + bw/2, oy + bd/2, -wt/2))
            self._spawn_prim(f"{ep}/WallNorth", north_cfg,
                             (ox + bw/2, oy + bd + wt/2, BIN_H/2))
            self._spawn_prim(f"{ep}/WallWest",  west_cfg,
                             (ox - wt/2, oy + bd/2, BIN_H/2))
            self._spawn_prim(f"{ep}/WallEast",  east_cfg,
                             (ox + bw + wt/2, oy + bd/2, BIN_H/2))

            # Kinematic pushers (parked outside bin)
            park_w = self._local_to_world(self._park, ei)
            self._spawn_prim(f"{ep}/PusherNS", pns_cfg, park_w)
            self._spawn_prim(f"{ep}/PusherEW", pew_cfg, park_w)

            # Dynamic objects — each spawned at a unique x position so they
            # don't interpenetrate before _place_objects teleports them.
            # Spread evenly along x, centred in y.
            n_total = 1 + self.n_obstacles  # target + obstacles
            step = max(OBJ_SIZE * 2.0, (bw - OBJ_SIZE) / max(n_total, 1))
            x0 = OBJ_SIZE
            self._spawn_prim(f"{ep}/Target", target_cfg,
                             self._local_to_world([x0, bd / 2, OBJ_H], ei))
            for oi in range(self.n_obstacles):
                self._spawn_prim(f"{ep}/Obstacle{oi}", obs_cfg,
                                 self._local_to_world([x0 + (oi + 1) * step,
                                                       bd / 2, OBJ_H], ei))

        # Wrap dynamic prims in batched RigidObject views.
        # spawn=None means "attach to existing prims, do not re-spawn".
        def _ro(pattern: str) -> RigidObject:
            return RigidObject(RigidObjectCfg(prim_path=pattern, spawn=None))

        self.pusher_ns_obj = _ro("/World/envs/env_.*/PusherNS")
        self.pusher_ew_obj = _ro("/World/envs/env_.*/PusherEW")
        self.target_obj    = _ro("/World/envs/env_.*/Target")
        self.obstacle_objs = [
            _ro(f"/World/envs/env_.*/Obstacle{i}")
            for i in range(self.n_obstacles)
        ]

        self.pusher_ns_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/PusherNS", history_length=1))
        self.pusher_ew_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/PusherEW", history_length=1))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _camera_view(self, ox: float, oy: float) -> tuple:
        """Return (eye, target) world positions scaled to the current bin size."""
        bw, bd = self.bin_w, self.bin_d
        view_dist = max(bw, bd) * 1.5
        target = (ox + bw / 2, oy + bd * 0.1, OBJ_H)
        # Camera direction: slightly right, mostly south, elevated (unit vector)
        eye = (target[0] + 0.183 * view_dist,
               target[1] - 0.948 * view_dist,
               target[2] + 0.320 * view_dist)
        return eye, target

    def _local_to_world(self, pos_local: list, env_idx: int) -> tuple:
        """Convert a bin-local 3-D position to world coordinates."""
        ox = self.env_origins[env_idx, 0].item()
        oy = self.env_origins[env_idx, 1].item()
        return (pos_local[0] + ox, pos_local[1] + oy, pos_local[2])

    def _world_to_local(self, pos_world: torch.Tensor, env_idx: int) -> torch.Tensor:
        """Subtract the env origin so positions are in bin-local frame."""
        origin = self.env_origins[env_idx].to(pos_world.device)
        return pos_world - origin

    # ------------------------------------------------------------------
    # Physics step helpers
    # ------------------------------------------------------------------

    def _refresh_all(self):
        """Pull fresh state from PhysX into every RigidObject's data cache."""
        dt = self.sim.get_physics_dt()
        for obj in ([self.pusher_ns_obj, self.pusher_ew_obj, self.target_obj]
                    + self.obstacle_objs):
            obj.update(dt)
        self.pusher_ns_sensor.update(dt)
        self.pusher_ew_sensor.update(dt)

    def _start_render_toggle_thread(self):
        """Spawn a daemon thread that reads raw input from /dev/tty and toggles rendering on 'f'."""
        import threading, tty, termios

        def _reader():
            try:
                with open('/dev/tty', 'rb', buffering=0) as tty_fh:
                    fd = tty_fh.fileno()
                    old = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        print('[IsaacLab] Press f to toggle rendering', flush=True)
                        while True:
                            ch = tty_fh.read(1)
                            if ch.lower() == b'f':
                                self.rendering_enabled = not self.rendering_enabled
                                print(f'\r[IsaacLab] Rendering {"ON" if self.rendering_enabled else "OFF"}', flush=True)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception as e:
                print(f'[IsaacLab] Render toggle unavailable: {e}', flush=True)

        threading.Thread(target=_reader, daemon=True).start()

    def _step_sim(self, render: bool = False):
        """Advance one physics step (all envs simultaneously) and refresh."""
        self.sim.step(render=(render or self._force_render) and self.rendering_enabled)
        self._refresh_all()
        if self._post_step_hook is not None:
            self._post_step_hook()

    # ------------------------------------------------------------------
    # Kinematic control helpers
    # ------------------------------------------------------------------

    def _set_pose(self, obj: RigidObject, pos_local: list,
                  quat_wxyz: tuple, env_ids: torch.Tensor):
        """
        Teleport a RigidObject to a bin-local position in the given envs.

        pos_local is in bin-local frame; internally converted to world frame
        for each env before writing to the physics backend.
        """
        n = len(env_ids)
        pos_world = torch.zeros(n, 3, device=self.device)
        for i, ei in enumerate(env_ids.tolist()):
            wp = self._local_to_world(pos_local, int(ei))
            pos_world[i] = torch.tensor(wp, device=self.device)
        quat = (torch.tensor(list(quat_wxyz), device=self.device)
                .unsqueeze(0).expand(n, -1))
        pose = torch.cat([pos_world, quat], dim=-1)   # (n, 7)
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        vel_zero = torch.zeros(n, 6, device=self.device)
        obj.write_root_velocity_to_sim(vel_zero, env_ids=env_ids)

    def _park_pushers(self, env_ids: torch.Tensor):
        """Park both pusher blades at the safe position for the given envs."""
        self._set_pose(self.pusher_ns_obj, self._park, _IDENTITY_QUAT, env_ids)
        self._set_pose(self.pusher_ew_obj, self._park, _IDENTITY_QUAT, env_ids)

    # ------------------------------------------------------------------
    # Object placement (single mode only)
    # ------------------------------------------------------------------

    def _place_objects(self):
        """Randomly place objects in env 0, settle, then save initial state.

        Objects are placed one at a time with a short settle between each so
        that PhysX commits each teleport before the next object is inserted.
        """
        from simulators.placement import random_initial_state
        self._refresh_all()

        state = random_initial_state(
            self.n_obstacles,
            stackable=self.stackable,
            difficult_spawn=self.difficult_spawn,
            bin_w=self.bin_w,
            bin_d=self.bin_d,
            debug=self.debug,
            n_z_levels=self.n_z_levels,
            target_z_level=self.target_z_level,
        )

        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        # Place obstacles first, then target; settle after each so PhysX
        # commits each teleport before the next object is inserted.
        for i, obj in enumerate(self.obstacle_objs):
            self._set_pose(obj, state['obstacle_pos'][i].tolist(), _IDENTITY_QUAT, env_ids)
            for _ in range(10):
                self._step_sim(render=self.show_viewer)

        self._set_pose(self.target_obj, state['target_pos'].tolist(), _IDENTITY_QUAT, env_ids)
        for _ in range(10):
            self._step_sim(render=self.show_viewer)

        self._park_pushers(env_ids)

        # Final settle with all objects in place
        for _ in range(60):
            self._step_sim(render=self.show_viewer)

        # Save as checkpoint for reset()
        self._initial_states = self._get_state(0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _get_state(self, env_idx: int) -> dict:
        """Read pos/quat of all objects from env_idx, in bin-local frame."""
        def _pos(obj: RigidObject) -> torch.Tensor:
            # root_pos_w shape: (n_envs, 3)
            return self._world_to_local(obj.data.root_pos_w[env_idx].clone(), env_idx)

        def _quat(obj: RigidObject) -> torch.Tensor:
            # root_quat_w shape: (n_envs, 4), convention (w,x,y,z)
            return obj.data.root_quat_w[env_idx].clone()

        return {
            'target_pos':    _pos(self.target_obj),
            'target_quat':   _quat(self.target_obj),
            'obstacle_pos':  torch.stack([_pos(o) for o in self.obstacle_objs]),
            'obstacle_quat': torch.stack([_quat(o) for o in self.obstacle_objs]),
        }

    def _set_state(self, state: dict, env_idx: int | None = None):
        """Teleport all objects to the given state, park pushers, zero velocities."""
        if env_idx is None:
            env_idx = 0
        env_ids = torch.tensor([env_idx], device=self.device, dtype=torch.long)

        def _lst(t) -> list:
            return t.tolist() if torch.is_tensor(t) else list(t)

        self._set_pose(self.target_obj,
                       _lst(state['target_pos']),
                       tuple(_lst(state['target_quat'])),
                       env_ids)
        for i, obs in enumerate(self.obstacle_objs):
            self._set_pose(obs,
                           _lst(state['obstacle_pos'][i]),
                           tuple(_lst(state['obstacle_quat'][i])),
                           env_ids)
        self._park_pushers(env_ids)

    def get_state(self, env_idx: int = 0) -> dict:
        """Public interface: get state from the given env slot."""
        return self._get_state(env_idx)

    def set_state(self, state: dict, env_idx: int | None = None):
        """Public interface: set state in env_idx (parallel) or env 0 (single)."""
        self._set_state(state, env_idx)

    def reset(self, seed: int | None = None, env_idx: int | None = None) -> dict:
        """
        Reset environment.
        - Single mode (n_envs=1): restore saved initial state (or re-randomise if seed given)
        - Parallel mode (n_envs>1): teleport one env slot to bin-centre initial state
        """
        if self.n_envs > 1:
            if seed is not None:
                torch.manual_seed(seed)
                self._place_objects()
                return self._get_state(0)
            if env_idx is not None:
                initial = {
                    'target_pos':    torch.tensor([self.bin_w/2, self.bin_d/2, OBJ_H],
                                                  device=self.device),
                    'target_quat':   torch.tensor([1., 0., 0., 0.],
                                                  device=self.device),
                    'obstacle_pos':  torch.tensor([[self.bin_w/2, self.bin_d/2, OBJ_H]],
                                                  device=self.device).expand(self.n_obstacles, -1),
                    'obstacle_quat': torch.tensor([[1., 0., 0., 0.]],
                                                  device=self.device).expand(self.n_obstacles, -1),
                }
                self._set_state(initial, env_idx)
                return self._get_state(env_idx)
            return {}
        else:
            if seed is not None:
                torch.manual_seed(seed)
                self._place_objects()
                return self._get_state(0)
            # Restore saved checkpoint
            if self._initial_states is not None:
                self._set_state(self._initial_states, 0)
                for _ in range(5):   # brief settle after teleport
                    self._step_sim(render=self.show_viewer)
            return self._get_state(0)

    # ------------------------------------------------------------------
    # Action primitives (single env)
    # ------------------------------------------------------------------

    def execute_ns_push(self, pos_2d: torch.Tensor, z: float,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0
                        ) -> tuple[dict, float, bool]:
        """Push northward: pusher_ns enters from south, sweeps north."""
        state = self._get_state(0)
        action = {'action_type': 'push_n', 'push_pos': pos_2d, 'push_z': z}
        return self.batch_evaluate([(state, action)])[0]

    def execute_ns_pull(self, pos_2d: torch.Tensor, z: float,
                        approach_dist: float = 0.12,
                        pull_steps: int | None = None, step_delay: float = 0.0
                        ) -> tuple[dict, float, bool]:
        """Pull southward: pusher_ns hooks north of object, sweeps south to exit."""
        state = self._get_state(0)
        action = {'action_type': 'pull_s', 'push_pos': pos_2d, 'push_z': z}
        return self.batch_evaluate([(state, action)])[0]

    def execute_ew_push(self, pos_2d: torch.Tensor, z: float, direction: int,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0
                        ) -> tuple[dict, float, bool]:
        """Push east (direction=+1) or west (direction=-1)."""
        state = self._get_state(0)
        action = {
            'action_type': 'push_e' if direction > 0 else 'push_w',
            'push_pos': pos_2d,
            'push_z': z,
        }
        return self.batch_evaluate([(state, action)])[0]

    # ------------------------------------------------------------------
    # Batch evaluation (parallel mode)
    # ------------------------------------------------------------------

    def _action_to_stroke(self, action: dict,
                          approach_dist: float = 0.12,
                          push_dist: float = 0.25) -> tuple[str, list, list]:
        """Convert action dict → (pusher_type, start_local_3d, end_local_3d)."""
        atype = action['action_type']
        pos   = action['push_pos']
        z     = action['push_z']
        if atype == 'push_n':
            return ('ns',
                    [pos[0], pos[1] - approach_dist, z],
                    [pos[0], pos[1] + push_dist,     z])
        if atype == 'pull_s':
            return ('ns',
                    [pos[0], pos[1] + approach_dist,  z],
                    [pos[0], EXIT_Y  - approach_dist, z])
        if atype == 'push_e':
            return ('ew',
                    [pos[0] - approach_dist, pos[1], z],
                    [pos[0] + push_dist,     pos[1], z])
        # push_w
        return ('ew',
                [pos[0] + approach_dist, pos[1], z],
                [pos[0] - push_dist,     pos[1], z])

    def _batch_evaluate_impl(self, pairs: list[tuple[dict, dict]]
                            ) -> list[tuple[dict, float, bool]]:
        """
        Evaluate up to n_envs (state, action) pairs in parallel.

        All envs advance together via a single sim.step() call per physics tick
        so wall-clock cost is roughly constant in k (GPU-bound).

        Parameters
        ----------
        pairs : list of (state_dict, action_dict)
            len(pairs) must be <= self.n_envs

        Returns
        -------
        list of (new_state, reward, done) — one per input pair
        """
        k = len(pairs)
        if k == 0:
            return []
        if k > self.n_envs:
            results = []
            for i in range(0, k, self.n_envs):
                results.extend(self._batch_evaluate_impl(pairs[i:i + self.n_envs]))
            return results

        # 1. Teleport each env to its starting state; compute stroke geometry
        strokes: list[tuple[str, list, list]] = []
        for env_idx, (state, action) in enumerate(pairs):
            self._set_state(state, env_idx)
            ptype, start, end = self._action_to_stroke(action)
            strokes.append((ptype, start, end))

        # Pre-allocate env_ids tensors to avoid GPU memory churn in loops
        env_ids_list = [torch.tensor([i], device=self.device, dtype=torch.long)
                        for i in range(k)]

        # Settle after teleporting objects (let solver resolve initial contacts)
        self._step_sim(render=False)
        self._step_sim(render=False)

        # 2. Warm-up: place pushers at stroke start, settle 2 ticks
        for env_idx, (ptype, start, _) in enumerate(strokes):
            obj = self.pusher_ns_obj if ptype == 'ns' else self.pusher_ew_obj
            self._set_pose(obj, start, _IDENTITY_QUAT, env_ids_list[env_idx])
        self._step_sim(render=self.show_viewer)
        self._step_sim(render=self.show_viewer)

        # 3. Sweep — one sim.step() advances ALL envs simultaneously
        force_stopped = [False] * k
        self.force_trace = []
        ptype0, _, _ = strokes[0]
        sensor0 = self.pusher_ns_sensor if ptype0 == 'ns' else self.pusher_ew_sensor
        for step_i in range(self.push_steps):
            t = (step_i + 1) / self.push_steps
            for env_idx, (ptype, start, end) in enumerate(strokes):
                if force_stopped[env_idx]:
                    continue
                pos = [s + t * (e - s) for s, e in zip(start, end)]
                obj = self.pusher_ns_obj if ptype == 'ns' else self.pusher_ew_obj
                self._set_pose(obj, pos, _IDENTITY_QUAT, env_ids_list[env_idx])
            self._step_sim(render=self.show_viewer)

            self.force_trace.append(float(sensor0.data.net_forces_w[0, 0].norm()))

            if self.force_threshold > 0:
                for env_idx, (ptype, start, end) in enumerate(strokes):
                    if force_stopped[env_idx]:
                        continue
                    sensor = self.pusher_ns_sensor if ptype == 'ns' else self.pusher_ew_sensor
                    force_mag = float(sensor.data.net_forces_w[env_idx, 0].norm())
                    if force_mag > self.force_threshold:
                        force_stopped[env_idx] = True
                        obj = self.pusher_ns_obj if ptype == 'ns' else self.pusher_ew_obj
                        self._set_pose(obj, self._park, _IDENTITY_QUAT, env_ids_list[env_idx])
                if all(force_stopped):
                    break

        # 4. Park all pushers and settle
        for env_idx in range(k):
            self._park_pushers(env_ids_list[env_idx])
        for _ in range(4):
            self._step_sim(render=self.show_viewer)

        # 5. Read back results
        results = []
        for env_idx in range(k):
            state  = self._get_state(env_idx)
            reward = self._compute_reward(state)
            done   = self._is_goal(state)
            results.append((state, reward, done))
        return results

    # ------------------------------------------------------------------
    # Reward / goal
    # ------------------------------------------------------------------

    def _obstacles_dropped(self, state: dict) -> bool:
        return any(float(state['obstacle_pos'][i][1]) < EXIT_Y
                   for i in range(self.n_obstacles))

    def _compute_reward(self, state: dict) -> float:
        cfg = self.reward_cfg
        r = 0.0
        if cfg is None or cfg.target_progress.enabled:
            y = float(state['target_pos'][1])
            r += float(torch.clamp(torch.tensor((self.bin_d / 2 - y) / (self.bin_d / 2 - EXIT_Y)), 0.0, 2.0))
        if cfg is None or cfg.obstacle_penalty.enabled:
            weight = 0.5 if cfg is None else cfg.obstacle_penalty.weight
            n_dropped = sum(1 for i in range(self.n_obstacles)
                            if float(state['obstacle_pos'][i][1]) < EXIT_Y)
            r -= weight * n_dropped
        if cfg is None or cfg.path_blocker.enabled:
            weight = 0.5 if cfg is None else cfg.path_blocker.weight
            scale  = 0.16 if cfg is None else cfg.path_blocker.scale
            tx = float(state['target_pos'][0])
            ty = float(state['target_pos'][1])
            for i in range(self.n_obstacles):
                oy = float(state['obstacle_pos'][i][1])
                if 0.0 < oy < ty:
                    x_dist = abs(float(state['obstacle_pos'][i][0]) - tx)
                    r -= weight * max(0.0, 1.0 - x_dist / scale)
        return r

    def _is_goal(self, state: dict) -> bool:
        if self._obstacles_dropped(state):
            return False
        return bool(float(state['target_pos'][1]) <= EXIT_Y)

    def is_goal(self, state: dict) -> bool:
        return self._is_goal(state)

    def step_physics(self) -> None:
        self._step_sim(render=False)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self, plan: list[dict], initial_state: dict) -> None:
        """Replay a plan in env 0 with the viewer open, using the same physics as planning."""
        ox = self.env_origins[0, 0].item()
        oy = self.env_origins[0, 1].item()
        eye, target = self._camera_view(ox, oy)
        self.sim.set_camera_view(eye=np.array(eye), target=np.array(target))

        self._set_state(initial_state, 0)
        self._force_render = True
        for _ in range(5):
            self._step_sim()

        input('Press Enter to start replay...')

        state = initial_state
        done  = False
        for step_i, action in enumerate(plan):
            atype = action['action_type']
            print(f'  Step {step_i + 1}/{len(plan)}: [{atype}] '
                  f'obj={action["obj_idx"]} '
                  f'pos={torch.round(action["push_pos"], decimals=3)} '
                  f'z={action["push_z"]:.3f}', end='', flush=True)
            (state, reward, done), = self.batch_evaluate([(state, action)])
            print(f'  -> reward={reward:.3f}, done={done}')
            if done:
                print('  Target escaped the bin!')
                break

        self._force_render = False
        if not done:
            print('  Plan executed (target may not have fully escaped).')
        input('Press Enter to close...')

    def record_replay(
        self,
        plan: list[dict],
        initial_state: dict,
        video_path: str,
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 30,
        capture_every: int = 4,
    ) -> str:
        """
        Replay a plan in env 0 and save a video, using the same physics as planning
        (including force-threshold stopping). Uses omni.replicator for headless capture.
        """
        import omni.replicator.core as rep
        import imageio

        ox = self.env_origins[0, 0].item()
        oy = self.env_origins[0, 1].item()
        bw, bd = self.bin_w, self.bin_d
        rec_target = (ox + bw / 2, oy + bd / 2, OBJ_H)
        rec_eye    = (ox + bw / 2, oy - 1.0,    1.2)
        camera = rep.create.camera(position=rec_eye, look_at=rec_target)
        render_product = rep.create.render_product(camera, resolution=resolution)
        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_ann.attach([render_product])

        self._set_state(initial_state, 0)
        frames: list = []
        frame_counter = [0]

        def _capture_hook():
            frame_counter[0] += 1
            if frame_counter[0] % capture_every == 0:
                self.sim.render()
                data = rgb_ann.get_data()
                if data is not None and data.size > 0:
                    frames.append(data[:, :, :3])  # RGBA → RGB

        self._post_step_hook = _capture_hook

        state = initial_state
        for action in plan:
            (state, _, done), = self.batch_evaluate([(state, action)])
            if done:
                break

        self._post_step_hook = None
        rgb_ann.detach()
        render_product.destroy()

        if not frames:
            print('  [record_replay] Warning: no frames captured, skipping video write.')
            return None

        print(f'  [record_replay] Captured {len(frames)} frames → {video_path}')
        imageio.mimwrite(video_path, frames, fps=fps, codec='libx264',
                         quality=8, macro_block_size=1)
        return video_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_target_pos_2d(self, state: dict | None = None,
                          env_idx: int = 0) -> torch.Tensor:
        if state is None:
            state = self._get_state(env_idx)
        return state['target_pos'][:2]

    def get_all_obj_positions_2d(self, state: dict | None = None,
                                 env_idx: int = 0) -> torch.Tensor:
        """Returns (N+1, 2) tensor: [target, obs0, obs1, ...]"""
        if state is None:
            state = self._get_state(env_idx)
        pos = [state['target_pos'][:2]]
        for i in range(self.n_obstacles):
            pos.append(state['obstacle_pos'][i][:2])
        return torch.stack(pos)
