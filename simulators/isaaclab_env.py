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

import colorsys
import logging
import math
import os
import tempfile
import time
import numpy as np
import torch
from typing import List, Optional
from tqdm import tqdm
from .base_env import SimulatorEnv
from .shapes import PIECES, shape_extents

logger = logging.getLogger(__name__)


def _obstacle_color(i: int, n: int) -> tuple:
    """Blue-family color for obstacle i of n, spread from cyan-blue to indigo-blue."""
    t = i / max(n - 1, 1)
    hue = 0.55 + t * 0.17
    sat = 0.65
    val = 0.95 - t * 0.20
    return colorsys.hsv_to_rgb(hue, sat, val)

try:
    import os as _os
    from isaaclab.app import AppLauncher
    _headless        = _os.environ.get('ISAACLAB_HEADLESS',        '0') == '1'
    _enable_cameras  = _os.environ.get('ISAACLAB_ENABLE_CAMERAS',  '0') == '1'
    app_launcher = AppLauncher(headless=_headless, enable_cameras=_enable_cameras)
    simulation_app = app_launcher.app


    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from isaaclab.assets import (RigidObject, RigidObjectCfg,
                                 RigidObjectCollection, RigidObjectCollectionCfg)
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        # Older package name (Isaac Lab < 2.0)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim import SimulationContext, SimulationCfg
        from omni.isaac.lab.assets import (RigidObject, RigidObjectCfg,
                                           RigidObjectCollection, RigidObjectCollectionCfg)
        from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
        ISAACLAB_AVAILABLE = True
    except ImportError:
        ISAACLAB_AVAILABLE = False
        sim_utils = None
        SimulationContext = None
        SimulationCfg = None
        RigidObject = None
        RigidObjectCfg = None
        RigidObjectCollection = None
        RigidObjectCollectionCfg = None
        ContactSensor = None
        ContactSensorCfg = None

# Bin dimensions (identical to genesis_env.py)
BIN_W  = 1.0    # x extent
BIN_D  = 1.0    # y extent (depth, from 0 to BIN_D)
BIN_H  = 0.5    # wall height
WALL_T = 0.05   # wall thickness

PUSHER_T = 0.012

EXIT_Y = -0.05   # target exits when its y < EXIT_Y

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
    dt : float
    seed : int | None
    stackable : bool
    friction : float
    n_z_levels : int
    push_steps : int
    substeps : int
    """

    def __init__(self, n_obstacles: int = 2, n_envs: int = 1,
                 dt: float = 0.01, seed: int | None = None,
                 stackable: bool = False, friction: float = 1.0,
                 n_z_levels: int = 1,
                 push_steps: int = 20, substeps: int = 4,
                 wall_thickness: float = WALL_T,
                 difficult_spawn: bool = False,
                 reward_cfg=None,
                 bin_size: float | None = None,
                 bin_size_factor: float = 0.9,
                 obj_size: float = 0.05,
                 obstacle_shapes: list[str] | None = None,
                 force_threshold: float = 100.0,
                 position_iterations: int = 4,
                 velocity_iterations: int = 1,
                 env_spacing_factor: float = 2.5,
                 post_teleport_steps: int = 10,
                 teleport_settle_steps: int = 3,
                 post_push_steps: int = 15,
                 debug: bool = False,
                 target_z_level: int | None = None,
                 force_obstacle_on_target: bool = False,
                 viewer_mode: str = 'replay'):
        if not ISAACLAB_AVAILABLE:
            raise ImportError(
                "isaaclab (or omni.isaac.lab) is not installed. "
                "Install Isaac Lab before using BinEnvIsaacLab."
            )

        _obs_shapes = (['cube'] * n_obstacles if obstacle_shapes is None
                       else list(obstacle_shapes))
        if len(_obs_shapes) < n_obstacles:
            _obs_shapes += ['cube'] * (n_obstacles - len(_obs_shapes))
        _obs_shapes = _obs_shapes[:n_obstacles]
        _max_cells = max(max(shape_extents(PIECES[s])) for s in _obs_shapes + ['cube'])
        _effective_size = _max_cells * obj_size
        if bin_size is None:
            bin_size = (n_obstacles + 1) * _effective_size * bin_size_factor

        super().__init__(
            n_obstacles=n_obstacles, n_envs=n_envs,
            dt=dt, seed=seed,
            stackable=stackable, friction=friction,
            n_z_levels=n_z_levels, push_steps=push_steps,
            substeps=substeps, wall_thickness=wall_thickness,
            difficult_spawn=difficult_spawn, reward_cfg=reward_cfg,
            bin_size=bin_size, bin_size_factor=bin_size_factor,
            obj_size=obj_size,
            debug=debug, target_z_level=target_z_level,
            force_obstacle_on_target=force_obstacle_on_target,
            viewer_mode=viewer_mode,
        )

        self.obstacle_shapes = _obs_shapes
        self._effective_obj_size = _effective_size

        self._OBJ_H    = self._OBJ_SIZE / 2
        self._pusher_w = self._OBJ_SIZE * 0.88

        self.force_threshold = force_threshold
        self.position_iterations = position_iterations
        self.velocity_iterations = velocity_iterations
        self.post_teleport_steps = post_teleport_steps
        self.teleport_settle_steps = teleport_settle_steps
        self.post_push_steps = post_push_steps
        self._in_push: bool = False  # slims _step_sim to sensor-only refresh during push loop
        self._active_push_sensors: list = []  # set before each push to only update needed sensors
        self._dbg_t_sim: float = 0.0
        self._dbg_t_sensors: float = 0.0
        self.force_trace: list[float] = []  # per-step force magnitudes for env 0, last batch_evaluate
        self._post_step_hook = None   # callable invoked after every _step_sim; used by record_replay
        self._force_render   = False  # when True, _step_sim always renders (used by replay)

        _park_y = -(max(self.bin_w, self.bin_d) * 1.5 + 0.1)
        self._park = [self.bin_w / 2, _park_y, self._OBJ_H]
        self.z_levels = [self._OBJ_H + i * self._OBJ_SIZE for i in range(n_z_levels)]
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"

        # Compute per-env world origins so envs don't overlap.
        # Account for wall thickness so thick walls don't cause env overlap.
        env_spacing = (max(self.bin_w, self.bin_d) + 2 * self.wall_thickness) * env_spacing_factor
        n_cols      = max(1, int(np.ceil(np.sqrt(self.n_envs))))
        self.env_origins = torch.zeros(self.n_envs, 3, device=self.device)
        for ei in range(self.n_envs):
            row = ei // n_cols
            col = ei % n_cols
            self.env_origins[ei, 0] = col * env_spacing
            self.env_origins[ei, 1] = row * env_spacing
        # CPU-side origin cache so _local_to_world avoids GPU→CPU sync
        self._env_origins_xy: list[tuple[float, float]] = [
            ((ei % n_cols) * env_spacing, (ei // n_cols) * env_spacing)
            for ei in range(self.n_envs)
        ]

        # Checkpoint for single-mode reset (saved after _place_objects settles)
        self._initial_states: dict | None = None

        # Rendering toggle: press 'f' in the viewer to flip this flag
        self.rendering_enabled: bool = True

        self._init_sim()

        # Pre-generate USD files for non-cube obstacle shapes.
        self._mesh_dir = tempfile.mkdtemp(prefix='puzzle_shapes_')
        self._obstacle_usd_paths: list[str | None] = []
        for shape_name in self.obstacle_shapes:
            if shape_name == 'cube':
                self._obstacle_usd_paths.append(None)
            else:
                usd_path = os.path.join(self._mesh_dir, f'{shape_name}.usda')
                if not os.path.exists(usd_path):
                    self._write_obstacle_usd(PIECES[shape_name], usd_path)
                self._obstacle_usd_paths.append(usd_path)

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
                min_position_iteration_count=self.position_iterations,
                min_velocity_iteration_count=self.velocity_iterations,
                enable_external_forces_every_iteration=True,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        logger.info('simulation device: %s', self.device)
        if not _headless:
            ox = self.env_origins[0, 0].item()
            oy = self.env_origins[0, 1].item()
            eye, target = self._camera_view(ox, oy)
            # set_camera_view is available in Isaac Lab >= 1.0
            self.sim.set_camera_view(
                eye=np.array(eye),
                target=np.array(target),
            )

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

    def _write_obstacle_usd(self, cells: list, path: str) -> None:
        """Write a USDA file for a multi-cell Tetris shape (compound rigid body)."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
        stage = Usd.Stage.CreateNew(path)
        root = UsdGeom.Xform.Define(stage, '/TetrisShape')
        stage.SetDefaultPrim(root.GetPrim())

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        cx = (max(rows) + min(rows)) / 2.0 * self._OBJ_SIZE
        cy = (max(cols) + min(cols)) / 2.0 * self._OBJ_SIZE

        for r, c in cells:
            cube = UsdGeom.Cube.Define(stage, f'/TetrisShape/cell_{r}_{c}')
            cube.GetSizeAttr().Set(self._OBJ_SIZE)
            cube.AddTranslateOp().Set(
                Gf.Vec3d(r * self._OBJ_SIZE - cx, c * self._OBJ_SIZE - cy, 0.0))
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)

        stage.Save()

    def _make_usd_cfg(self, usd_path: str, mass: float, color: tuple,
                      pos_iters: int | None = None):
        """Return a UsdFileCfg for spawning a multi-cell compound obstacle."""
        vis = (sim_utils.PreviewSurfaceCfg(diffuse_color=color)
               if color is not None else None)
        return sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=pos_iters,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.friction,
                dynamic_friction=self.friction,
                restitution=0.0,
            ),
            visual_material=vis,
        )

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
            (self._pusher_w, PUSHER_T, self._pusher_w),
            mass=10.0, kinematic=True, friction=fr, color=(0.2, 0.9, 0.2),
            activate_contact_sensors=True)
        pew_cfg    = self._make_box_cfg(
            (PUSHER_T, self._pusher_w, self._pusher_w),
            mass=10.0, kinematic=True, friction=fr, color=(0.9, 0.6, 0.1),
            activate_contact_sensors=True)
        target_cfg = self._make_box_cfg(
            (self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE),
            mass=0.05, kinematic=False, friction=fr, color=(0.9, 0.2, 0.2),
            pos_iters=8)
        obs_cfgs = []
        for oi in range(self.n_obstacles):
            usd_path = self._obstacle_usd_paths[oi]
            color = _obstacle_color(oi, self.n_obstacles)
            if usd_path is None:
                obs_cfgs.append(self._make_box_cfg(
                    (self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE),
                    mass=0.5, kinematic=False, friction=fr,
                    color=color, pos_iters=8))
            else:
                obs_cfgs.append(self._make_usd_cfg(
                    usd_path, mass=0.5, color=color, pos_iters=8))

        # Visual-only plane marking EXIT_Y — kinematic, collision disabled so
        # the target can pass through it freely.
        exit_marker_cfg = sim_utils.CuboidCfg(
            size=(bw, 0.004, .02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        )

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

            # Exit boundary marker at EXIT_Y (south of the bin opening)
            self._spawn_prim(f"{ep}/ExitMarker", exit_marker_cfg,
                             (ox + bw/2, oy + EXIT_Y, self._OBJ_H))

            # Kinematic pushers (parked outside bin)
            park_w = self._local_to_world(self._park, ei)
            self._spawn_prim(f"{ep}/PusherNS", pns_cfg, park_w)
            self._spawn_prim(f"{ep}/PusherEW", pew_cfg, park_w)

            # Dynamic objects — each spawned at a unique x position so they
            # don't interpenetrate before _place_objects teleports them.
            # Spread evenly along x, centred in y.
            n_total = 1 + self.n_obstacles  # target + obstacles
            step = max(self._OBJ_SIZE * 2.0, (bw - self._OBJ_SIZE) / max(n_total, 1))
            x0 = self._OBJ_SIZE
            self._spawn_prim(f"{ep}/Target", target_cfg,
                             self._local_to_world([x0, bd / 2, self._OBJ_H], ei))
            for oi in range(self.n_obstacles):
                self._spawn_prim(f"{ep}/Obstacle{oi}", obs_cfgs[oi],
                                 self._local_to_world([x0 + (oi + 1) * step,
                                                       bd / 2, self._OBJ_H], ei))

        # Wrap dynamic prims in batched RigidObject views.
        # spawn=None means "attach to existing prims, do not re-spawn".
        def _ro(pattern: str) -> RigidObject:
            return RigidObject(RigidObjectCfg(prim_path=pattern, spawn=None))

        self.pusher_ns_obj = _ro("/World/envs/env_.*/PusherNS")
        self.pusher_ew_obj = _ro("/World/envs/env_.*/PusherEW")
        self.target_obj    = _ro("/World/envs/env_.*/Target")
        self.obstacle_collection = RigidObjectCollection(
            RigidObjectCollectionCfg(rigid_objects={
                f"obs{i}": RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Obstacle{i}", spawn=None
                )
                for i in range(self.n_obstacles)
            })
        )

        self.pusher_ns_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/PusherNS", history_length=1))
        self.pusher_ew_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/PusherEW", history_length=1))

        # Pre-allocated single-row buffers for kinematic pusher updates.
        # Used by _set_pose for n=1 calls outside the push loop (warm-up, park, _set_state).
        self._pose_buf = {
            self.pusher_ns_obj: torch.zeros(1, 7, device=self.device),
            self.pusher_ew_obj: torch.zeros(1, 7, device=self.device),
        }
        self._vel_buf = {
            self.pusher_ns_obj: torch.zeros(1, 6, device=self.device),
            self.pusher_ew_obj: torch.zeros(1, 6, device=self.device),
        }
        _iq = torch.tensor(list(_IDENTITY_QUAT), device=self.device)
        for buf in self._pose_buf.values():
            buf[0, 3:] = _iq

        # Batch-write buffers for the push loop. Instead of k separate
        # write_root_pose_to_sim calls (each with fixed CUDA-sync overhead),
        # we write all active envs in one call per pusher per step.
        self._batch_pose_ns  = torch.zeros(self.n_envs, 7, device=self.device)
        self._batch_pose_ew  = torch.zeros(self.n_envs, 7, device=self.device)
        self._batch_vel_zero = torch.zeros(self.n_envs, 6, device=self.device)
        self._batch_ids_ns   = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self._batch_ids_ew   = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self._batch_pose_ns[:, 3:] = _iq
        self._batch_pose_ew[:, 3:] = _iq

        # Pre-allocated single-row buffer for target pose writes in _set_state.
        self._tgt_pose_buf = torch.zeros(1, 7, device=self.device)
        self._tgt_vel_buf  = torch.zeros(1, 6, device=self.device)

        # Batch state-set buffers for _set_state_batch (used by _batch_evaluate_impl).
        # (n_envs, 3) world origins — avoids per-env Python loops on the hot path.
        self._origins_xyz = torch.tensor(
            [[ox, oy, 0.0] for ox, oy in self._env_origins_xy],
            dtype=torch.float32, device=self.device)
        # (n_envs, 7) target pose buffer; quat cols pre-filled with identity.
        self._batch_tgt_pose_buf = torch.zeros(self.n_envs, 7, dtype=torch.float32, device=self.device)
        self._batch_tgt_pose_buf[:, 3:] = _iq
        # (n_envs, n_obstacles, 13) obstacle state buffer (pos+quat+vel); vel stays zero.
        self._batch_obs_state_buf = torch.zeros(
            self.n_envs, self.n_obstacles, 13, dtype=torch.float32, device=self.device)
        # Pre-built park poses in world frame for every env (n_envs, 7).
        _park_t = torch.tensor(self._park, dtype=torch.float32, device=self.device)
        self._batch_park_pose_ns = torch.zeros(self.n_envs, 7, dtype=torch.float32, device=self.device)
        self._batch_park_pose_ew = torch.zeros(self.n_envs, 7, dtype=torch.float32, device=self.device)
        self._batch_park_pose_ns[:, :3] = self._origins_xyz + _park_t
        self._batch_park_pose_ew[:, :3] = self._origins_xyz + _park_t
        self._batch_park_pose_ns[:, 3:] = _iq
        self._batch_park_pose_ew[:, 3:] = _iq

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _camera_view(self, ox: float, oy: float) -> tuple:
        """Return (eye, target) world positions scaled to the current bin size."""
        bw, bd = self.bin_w, self.bin_d
        view_dist = max(bw, bd) * 1.5
        target = (ox + bw / 2, oy + bd * 0.1, self._OBJ_H)
        # Camera direction: slightly right, mostly south, elevated (unit vector)
        eye = (target[0] + 0.183 * view_dist,
               target[1] - 0.948 * view_dist,
               target[2] + 0.320 * view_dist)
        return eye, target

    def _local_to_world(self, pos_local: list, env_idx: int) -> tuple:
        """Convert a bin-local 3-D position to world coordinates."""
        ox, oy = self._env_origins_xy[env_idx]
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
        for obj in [self.pusher_ns_obj, self.pusher_ew_obj, self.target_obj]:
            obj.update(dt)
        self.obstacle_collection.update(dt)
        self.pusher_ns_sensor.update(dt)
        self.pusher_ew_sensor.update(dt)

    def _refresh_sensors_only(self):
        """Refresh only the force sensors that are active in the current push batch."""
        dt = self.sim.get_physics_dt()
        for sensor in self._active_push_sensors:
            sensor.update(dt)

    def _step_sim(self, render: bool = False):
        """Advance one physics step (all envs simultaneously) and refresh."""
        if self.debug and self._in_push:
            _t0 = time.perf_counter()
            self.sim.step(render=(render or self._force_render) and self.rendering_enabled)
            self._dbg_t_sim += time.perf_counter() - _t0
            _t0 = time.perf_counter()
            self._refresh_sensors_only()
            self._dbg_t_sensors += time.perf_counter() - _t0
        else:
            self.sim.step(render=(render or self._force_render) and self.rendering_enabled)
            if self._in_push:
                self._refresh_sensors_only()
            else:
                self._refresh_all()
        if self._post_step_hook is not None:
            self._post_step_hook()

    # ------------------------------------------------------------------
    # Kinematic control helpers
    # ------------------------------------------------------------------

    def _set_pose(self, obj: RigidObject, pos_local: list,
                  quat_wxyz: tuple, env_ids: torch.Tensor,
                  env_idx: int | None = None):
        """
        Teleport a RigidObject to a bin-local position in the given envs.

        pos_local is in bin-local frame; internally converted to world frame
        for each env before writing to the physics backend.
        Pass env_idx when known to avoid env_ids[0].item() GPU→CPU sync.
        """
        n = len(env_ids)
        pose_buf = self._pose_buf.get(obj) if n == 1 else None
        # _pose_buf keys are the kinematic pushers; PhysX rejects velocity
        # writes on kinematic bodies, so skip them here and in the slow path.
        is_kinematic = obj in self._pose_buf
        if pose_buf is not None:
            # Fast path: reuse pre-allocated buffers (n=1, pusher hot loop)
            # Quat is pre-filled with _IDENTITY_QUAT at build time — no write needed.
            ei = env_idx if env_idx is not None else int(env_ids[0].item())
            ox, oy = self._env_origins_xy[ei]
            pose_buf[0, 0] = pos_local[0] + ox
            pose_buf[0, 1] = pos_local[1] + oy
            pose_buf[0, 2] = pos_local[2]
            obj.write_root_pose_to_sim(pose_buf, env_ids=env_ids)
        else:
            pos_world = torch.zeros(n, 3, device=self.device)
            for i, ei in enumerate(env_ids.tolist()):
                wp = self._local_to_world(pos_local, int(ei))
                pos_world[i] = torch.tensor(wp, device=self.device)
            quat = (torch.tensor(list(quat_wxyz), device=self.device)
                    .unsqueeze(0).expand(n, -1))
            pose = torch.cat([pos_world, quat], dim=-1)
            obj.write_root_pose_to_sim(pose, env_ids=env_ids)
            if not is_kinematic:
                vel_zero = torch.zeros(n, 6, device=self.device)
                obj.write_root_velocity_to_sim(vel_zero, env_ids=env_ids)

    def _park_pushers(self, env_ids: torch.Tensor, env_idx: int | None = None):
        """Park both pusher blades at the safe position for the given envs."""
        self._set_pose(self.pusher_ns_obj, self._park, _IDENTITY_QUAT, env_ids, env_idx=env_idx)
        self._set_pose(self.pusher_ew_obj, self._park, _IDENTITY_QUAT, env_ids, env_idx=env_idx)

    def wait_for_input(self, prompt: str = '  [Press Enter to continue...]') -> None:
        """Keep the Isaac Sim viewport live while waiting for the user to press Enter.

        Spins on sim.render() + non-blocking stdin poll so the viewer stays
        interactive (plain input() would freeze the UI thread).
        """
        import sys
        import select
        print(prompt, flush=True)
        while True:
            self.sim.render()
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                sys.stdin.readline()
                break

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
            obj_size=self._effective_obj_size,
            stackable=self.stackable,
            difficult_spawn=self.difficult_spawn,
            bin_w=self.bin_w,
            bin_d=self.bin_d,
            debug=self.debug,
            n_z_levels=self.n_z_levels,
            target_z_level=self.target_z_level,
            force_obstacle_on_target=self.force_obstacle_on_target,
            obj_height=self._OBJ_H,
        )

        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        # Place obstacles first, then target; settle after each so PhysX
        # commits each teleport before the next object is inserted.
        for i in range(self.n_obstacles):
            pos_w  = torch.tensor(self._local_to_world(state['obstacle_pos'][i].tolist(), 0),
                                  device=self.device).unsqueeze(0)
            quat_t = torch.tensor(list(_IDENTITY_QUAT), device=self.device).unsqueeze(0)
            vel_t  = torch.zeros(1, 6, device=self.device)
            obs_state = torch.cat([pos_w, quat_t, vel_t], dim=-1).unsqueeze(0)
            self.obstacle_collection.write_object_state_to_sim(
                obs_state, env_ids=env_ids,
                object_ids=torch.tensor([i], device=self.device),
            )
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

        # Warn if any obstacle is significantly tilted after settling
        quats = self._initial_states['obstacle_quat']   # (n_obs, 4) wxyz
        dot = quats[:, 0].abs()                          # |w| component; 1.0 = identity
        tilt_deg = 2.0 * torch.acos(dot.clamp(max=1.0)) * (180.0 / math.pi)
        bad = tilt_deg > 1.0
        if bad.any():
            for i in bad.nonzero(as_tuple=True)[0].tolist():
                logger.warning(
                    f'_place_objects: obstacle {i} has {tilt_deg[i].item():.1f}° tilt '
                    f'after settle (quat={quats[i].tolist()})'
                )

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
            'obstacle_pos':  (self.obstacle_collection.data.object_link_pose_w[env_idx, :, :3].clone()
                              - self.env_origins[env_idx].to(self.device)),
            'obstacle_quat': self.obstacle_collection.data.object_link_pose_w[env_idx, :, 3:].clone(),
        }

    def _set_state(self, state: dict, env_idx: int | None = None,
                   env_ids: torch.Tensor | None = None):
        """Teleport all objects to the given state, park pushers, zero velocities."""
        if env_idx is None:
            env_idx = 0
        if env_ids is None:
            env_ids = torch.tensor([env_idx], device=self.device, dtype=torch.long)
        self._set_state_batch([state], env_ids)

    def _set_state_batch(self, states: list[dict],
                         env_ids: torch.Tensor):
        """Teleport k envs to k states in 6 batched API calls instead of k×6 sequential calls."""
        k = len(states)
        origins = self._origins_xyz[env_ids]           # (k, 3), pure GPU index

        # Target
        tgt_pos  = torch.stack([s['target_pos'].to(self.device)  for s in states])  # (k, 3)
        tgt_quat = torch.stack([s['target_quat'].to(self.device) for s in states])  # (k, 4)
        buf = self._batch_tgt_pose_buf[:k]
        buf[:, :3] = tgt_pos + origins
        buf[:, 3:]  = tgt_quat
        self.target_obj.write_root_pose_to_sim(buf, env_ids=env_ids)
        self.target_obj.write_root_velocity_to_sim(self._batch_vel_zero[:k], env_ids=env_ids)

        # Obstacles
        obs_pos  = torch.stack([s['obstacle_pos'].to(self.device)  for s in states])  # (k, n_obs, 3)
        obs_quat = torch.stack([s['obstacle_quat'].to(self.device) for s in states])  # (k, n_obs, 4)
        obs_buf  = self._batch_obs_state_buf[:k]
        obs_buf[:, :, :3]  = obs_pos + origins.unsqueeze(1)
        obs_buf[:, :, 3:7] = obs_quat
        # obs_buf[:, :, 7:] stays zero (pre-zeroed at alloc time)
        self.obstacle_collection.write_object_state_to_sim(obs_buf, env_ids=env_ids)

        # Pushers — park poses pre-built per env, no Python loop.
        # Pushers are kinematic; PhysX rejects velocity writes on them.
        self.pusher_ns_obj.write_root_pose_to_sim(self._batch_park_pose_ns[:k], env_ids=env_ids)
        self.pusher_ew_obj.write_root_pose_to_sim(self._batch_park_pose_ew[:k], env_ids=env_ids)

    def get_state(self, env_idx: int = 0) -> dict:
        """Public interface: get state from the given env slot."""
        return self._get_state(env_idx)

    def set_state(self, state: dict, env_idx: int | None = None):
        """Public interface: set state in env_idx (parallel) or env 0 (single)."""
        self._set_state(state, env_idx)

    def close(self):
        """Cleanly tear down the simulation so simulation_app.close() doesn't hang."""
        if not hasattr(self, 'sim') or self.sim is None:
            return
        if not self.sim.is_stopped():
            self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

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
                    'target_pos':    torch.tensor([self.bin_w/2, self.bin_d/2, self._OBJ_H],
                                                  device=self.device),
                    'target_quat':   torch.tensor([1., 0., 0., 0.],
                                                  device=self.device),
                    'obstacle_pos':  torch.tensor([[self.bin_w/2, self.bin_d/2, self._OBJ_H]],
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
        env_ids_list = [torch.tensor([i], device=self.device, dtype=torch.long)
                        for i in range(k)]
        env_ids_k = torch.arange(k, device=self.device, dtype=torch.long)
        states_k  = [s for s, _ in pairs]
        strokes: list[tuple[str, list, list]] = []
        self._set_state_batch(states_k, env_ids_k)
        for _, action in pairs:
            ptype, start, end = self._action_to_stroke(action)
            strokes.append((ptype, start, end))

        # Repeated-teleport settle: re-write all objects to target positions after every
        # step so the contact manifold builds up against the correct configuration rather
        # than stale PhysX warm-start impulses from the prior frame.
        for _ in range(self.teleport_settle_steps):
            self._step_sim(render=False)
            self._set_state_batch(states_k, env_ids_k)

        # Free settle — lets objects find their resting contact.
        for _ in range(self.post_teleport_steps):
            self._step_sim(render=False)

        # 2. Warm-up: place pushers at stroke start, settle 2 ticks
        for env_idx, (ptype, start, _) in enumerate(strokes):
            obj = self.pusher_ns_obj if ptype == 'ns' else self.pusher_ew_obj
            self._set_pose(obj, start, _IDENTITY_QUAT, env_ids_list[env_idx], env_idx=env_idx)
        self._step_sim(render=self.show_viewer)
        self._step_sim(render=self.show_viewer)

        # 3. Sweep — one sim.step() advances ALL envs simultaneously

        _force_buf: list[torch.Tensor] = []
        ptype0, _, _ = strokes[0]
        sensor0 = self.pusher_ns_sensor if ptype0 == 'ns' else self.pusher_ew_sensor
        ptypes_used = {ptype for ptype, _, _ in strokes}
        self._active_push_sensors = (
            [self.pusher_ns_sensor] * ('ns' in ptypes_used) +
            [self.pusher_ew_sensor] * ('ew' in ptypes_used)
        )
        self._in_push = True
        self._dbg_t_sim = self._dbg_t_sensors = 0.0
        _t_loop = _t_write = _t_step = 0.0

        # Pre-compute per-pusher-type start/delta tensors (with world origins baked in)
        # so the inner push loop can be two tensor ops instead of k scalar Python writes.
        _ns_idxs = [i for i, (pt, _, _) in enumerate(strokes) if pt == 'ns']
        _ew_idxs = [i for i, (pt, _, _) in enumerate(strokes) if pt == 'ew']

        def _stroke_tensors(idxs):
            if not idxs:
                z = torch.empty(0, 3, dtype=torch.float32, device=self.device)
                return z, z, torch.empty(0, dtype=torch.long, device=self.device)
            starts = torch.tensor([strokes[i][1] for i in idxs], dtype=torch.float32)
            ends   = torch.tensor([strokes[i][2] for i in idxs], dtype=torch.float32)
            ox     = torch.tensor([self._env_origins_xy[i][0] for i in idxs], dtype=torch.float32)
            oy     = torch.tensor([self._env_origins_xy[i][1] for i in idxs], dtype=torch.float32)
            pos0   = starts.clone(); pos0[:, 0] += ox; pos0[:, 1] += oy
            delta  = ends - starts   # bin-local delta; origin cancels in the difference
            ids    = torch.tensor(idxs, dtype=torch.long)
            return pos0.to(self.device), delta.to(self.device), ids.to(self.device)

        _ns_pos0, _ns_delta, _ns_ids = _stroke_tensors(_ns_idxs)
        _ew_pos0, _ew_delta, _ew_ids = _stroke_tensors(_ew_idxs)
        _n_ns, _n_ew = len(_ns_idxs), len(_ew_idxs)
        if _n_ns: self._batch_ids_ns[:_n_ns] = _ns_ids
        if _n_ew: self._batch_ids_ew[:_n_ew] = _ew_ids
        # Active slices — start as the full arrays, rebuilt only on force-stop events
        _ns_pos0_a, _ns_delta_a, _ns_ids_a, n_ns_a = _ns_pos0, _ns_delta, _ns_ids, _n_ns
        _ew_pos0_a, _ew_delta_a, _ew_ids_a, n_ew_a = _ew_pos0, _ew_delta, _ew_ids, _n_ew
        _ns_active = torch.ones(_n_ns, dtype=torch.bool, device=self.device)
        _ew_active = torch.ones(_n_ew, dtype=torch.bool, device=self.device)
        _ns_env_to_pos = {env: p for p, env in enumerate(_ns_idxs)}
        _ew_env_to_pos = {env: p for p, env in enumerate(_ew_idxs)}
        _force_threshold_t = torch.tensor(self.force_threshold, device=self.device)

        for step_i in tqdm(range(self.push_steps), desc='push', leave=False):
            t = (step_i + 1) / self.push_steps
            _t0 = time.perf_counter()
            if n_ns_a:
                self._batch_pose_ns[:n_ns_a, :3] = _ns_pos0_a + t * _ns_delta_a
            if n_ew_a:
                self._batch_pose_ew[:n_ew_a, :3] = _ew_pos0_a + t * _ew_delta_a
            _t_loop += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            if n_ns_a:
                self.pusher_ns_obj.write_root_pose_to_sim(
                    self._batch_pose_ns[:n_ns_a], env_ids=self._batch_ids_ns[:n_ns_a])
            if n_ew_a:
                self.pusher_ew_obj.write_root_pose_to_sim(
                    self._batch_pose_ew[:n_ew_a], env_ids=self._batch_ids_ew[:n_ew_a])
            _t_write += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            self._step_sim(render=self.show_viewer)
            _t_step += time.perf_counter() - _t0

            _force_buf.append(sensor0.data.net_forces_w[0, 0].norm())

            if self.force_threshold > 0:
                # Vectorised force check: all comparisons stay on GPU; only one
                # .any() sync per pusher type per step (vs k .item() syncs before).
                # The inner loop only runs on actual stop events (rare).
                if n_ns_a:
                    ns_over = (self.pusher_ns_sensor.data.net_forces_w[_ns_ids_a, 0]
                               .norm(dim=-1) > _force_threshold_t)
                    for local_i in ns_over.nonzero(as_tuple=True)[0].tolist():
                        env_idx = int(_ns_ids_a[local_i])
                        self._set_pose(self.pusher_ns_obj, self._park, _IDENTITY_QUAT,
                                       env_ids_list[env_idx], env_idx=env_idx)
                        _ns_active[_ns_env_to_pos[env_idx]] = False
                    if not _ns_active.all():
                        _ns_pos0_a  = _ns_pos0[_ns_active]
                        _ns_delta_a = _ns_delta[_ns_active]
                        _ns_ids_a   = _ns_ids[_ns_active]
                        n_ns_a = int(_ns_active.sum())
                        if n_ns_a: self._batch_ids_ns[:n_ns_a] = _ns_ids_a
                if n_ew_a:
                    ew_over = (self.pusher_ew_sensor.data.net_forces_w[_ew_ids_a, 0]
                               .norm(dim=-1) > _force_threshold_t)
                    for local_i in ew_over.nonzero(as_tuple=True)[0].tolist():
                        env_idx = int(_ew_ids_a[local_i])
                        self._set_pose(self.pusher_ew_obj, self._park, _IDENTITY_QUAT,
                                       env_ids_list[env_idx], env_idx=env_idx)
                        _ew_active[_ew_env_to_pos[env_idx]] = False
                    if not _ew_active.all():
                        _ew_pos0_a  = _ew_pos0[_ew_active]
                        _ew_delta_a = _ew_delta[_ew_active]
                        _ew_ids_a   = _ew_ids[_ew_active]
                        n_ew_a = int(_ew_active.sum())
                        if n_ew_a: self._batch_ids_ew[:n_ew_a] = _ew_ids_a
                if n_ns_a == 0 and n_ew_a == 0:
                    break
        self._in_push = False
        self.force_trace = torch.stack(_force_buf).cpu().tolist()
        if self.debug:
            n_steps = len(self.force_trace)
            logger.debug(
                'push %d steps × %d envs | loop=%.1fms  write=%.1fms  '
                'step=%.1fms (sim=%.1fms sensors=%.1fms)  total=%.1fms',
                n_steps, k,
                _t_loop * 1000, _t_write * 1000, _t_step * 1000,
                self._dbg_t_sim * 1000, self._dbg_t_sensors * 1000,
                (_t_loop + _t_write + _t_step) * 1000,
            )

        # 4. Park all pushers and settle
        for env_idx in range(k):
            self._park_pushers(env_ids_list[env_idx], env_idx=env_idx)
        for _ in range(self.post_push_steps):
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

    def _is_goal(self, state: dict) -> bool:
        if self._obstacles_dropped(state):
            return False
        return bool(float(state['target_pos'][1]) <= EXIT_Y)

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
        self._force_render = self.viewer_mode != 'headless'
        for _ in range(5):
            self._step_sim()

        input('Press Enter to start replay...')

        state = initial_state
        done  = False
        for step_i, action in enumerate(plan):
            atype = action['action_type']
            logger.info('  Step %d/%d: [%s] obj=%s pos=%s z=%.3f',
                        step_i + 1, len(plan), atype, action["obj_idx"],
                        torch.round(action["push_pos"], decimals=3), action["push_z"])
            (state, reward, done), = self.batch_evaluate([(state, action)])
            logger.info('  -> reward=%.3f, done=%s', reward, done)
            if done:
                logger.info('  Target escaped the bin!')
                break

        self._force_render = False
        if not done:
            logger.info('  Plan executed (target may not have fully escaped).')
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
        rec_target = (ox + bw / 2, oy + bd / 2, self._OBJ_H)
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
            logger.warning('record_replay: no frames captured, skipping video write.')
            return None

        logger.info('record_replay: captured %d frames → %s', len(frames), video_path)
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
