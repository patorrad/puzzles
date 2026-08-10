"""
Push Prediction Network (PPN) for MORE (Huang et al., ICRA 2022).

Two architecture options, both selected via ``build_ppn(arch, n_obstacles)``:

``arch='deepsets'`` — ``PPN``
    DeepSets over the object set (permutation-invariant over N obstacles)
    combined with a push encoding.

``arch='mlp'`` — ``PPNFlat``
    Flat MLP matching AlphaZero's SolverNet style.  Section-contiguous
    scene encoding (target_xyz + obs_xyz + target_quat + obs_quat) concatenated
    with push features, fed through a 2-layer trunk.

Both classes expose the same interface::

    forward(states_dict, pushes_dict) → (B,)
    forward_single(state_dict, push_dict) → scalar

Use ``build_ppn`` to construct either architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.encoders import _canonical_quat


class PPN(nn.Module):
    """
    Push Prediction Network.

    Parameters
    ----------
    n_obstacles : int
    obj_emb_dim : int
        Hidden dimension for the per-object encoder φ.
    push_emb_dim : int
        Hidden dimension for the push encoder ψ.
    agg_hidden : int
        Hidden dim of the aggregation MLP.
    """

    OBJ_INPUT_DIM  = 8   # xyz(3) + quat(4) + is_target(1)
    PUSH_INPUT_DIM = 5   # start_xy(2) + dir_xy(2) + z(1)

    def __init__(self, n_obstacles: int,
                 obj_emb_dim:  int = 64,
                 push_emb_dim: int = 64,
                 agg_hidden:   int = 128):
        super().__init__()
        self.n_obstacles = n_obstacles

        # φ — per-object encoder
        self.phi = nn.Sequential(
            nn.Linear(self.OBJ_INPUT_DIM, obj_emb_dim), nn.ReLU(),
            nn.Linear(obj_emb_dim, obj_emb_dim), nn.ReLU(),
        )

        # ψ — push encoder
        self.psi = nn.Sequential(
            nn.Linear(self.PUSH_INPUT_DIM, push_emb_dim), nn.ReLU(),
            nn.Linear(push_emb_dim, push_emb_dim), nn.ReLU(),
        )

        # Combined: max-pool + sum-pool → 2*obj_emb_dim; cat with push_emb_dim
        combined_dim = 2 * obj_emb_dim + push_emb_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, agg_hidden), nn.ReLU(),
            nn.Linear(agg_hidden, 1),
        )

    # ------------------------------------------------------------------
    # Batch forward
    # ------------------------------------------------------------------

    def forward(self, states: dict, pushes: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        states : dict with batched tensors
            'target_pos'    : (B, 3)
            'target_quat'   : (B, 4)
            'obstacle_pos'  : (B, N, 3)
            'obstacle_quat' : (B, N, 4)
        pushes : dict with batched tensors
            'push_start_xy' : (B, 2)
            'push_end_xy'   : (B, 2)
            'push_z'        : (B,) or (B, 1)

        Returns
        -------
        q : (B,) Q-value predictions
        """
        obj_enc = self._encode_objects(states)        # (B, 2*obj_emb_dim)
        push_enc = self._encode_push(pushes)          # (B, push_emb_dim)
        x = torch.cat([obj_enc, push_enc], dim=-1)    # (B, combined_dim)
        return self.head(x).squeeze(-1)               # (B,)

    # ------------------------------------------------------------------
    # Single-sample forward (no batching overhead in tree search)
    # ------------------------------------------------------------------

    def forward_single(self, state: dict, push: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        state : dict with unbatched tensors
            'target_pos'    : (3,)
            'target_quat'   : (4,)
            'obstacle_pos'  : (N, 3)
            'obstacle_quat' : (N, 4)
        push : dict with unbatched tensors
            'push_start_xy' : (2,)
            'push_end_xy'   : (2,)
            'push_z'        : scalar tensor or float

        Returns
        -------
        q : scalar tensor
        """
        batched_state = {
            'target_pos':    state['target_pos'].unsqueeze(0),
            'target_quat':   state['target_quat'].unsqueeze(0),
            'obstacle_pos':  state['obstacle_pos'].unsqueeze(0),
            'obstacle_quat': state['obstacle_quat'].unsqueeze(0),
        }
        batched_push = {
            'push_start_xy': push['push_start_xy'].unsqueeze(0),
            'push_end_xy':   push['push_end_xy'].unsqueeze(0),
            'push_z':        torch.as_tensor(push['push_z']).float().unsqueeze(0),
        }
        return self.forward(batched_state, batched_push).squeeze(0)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _encode_objects(self, states: dict) -> torch.Tensor:
        """DeepSets over the (target + obstacle) set → (B, 2*obj_emb_dim)."""
        B = states['target_pos'].shape[0]

        # Build (B, N+1, OBJ_INPUT_DIM) token matrix
        target_token = torch.cat([
            states['target_pos'].float(),          # (B, 3)
            states['target_quat'].float(),         # (B, 4)
            torch.ones(B, 1, device=states['target_pos'].device),
        ], dim=-1).unsqueeze(1)                    # (B, 1, 8)

        obs_pos  = states['obstacle_pos'].float()   # (B, N, 3)
        obs_quat = states['obstacle_quat'].float()  # (B, N, 4)
        N = obs_pos.shape[1]
        obs_token = torch.cat([
            obs_pos,
            obs_quat,
            torch.zeros(B, N, 1, device=obs_pos.device),
        ], dim=-1)                                  # (B, N, 8)

        tokens = torch.cat([target_token, obs_token], dim=1)  # (B, N+1, 8)

        # φ applied to each token
        emb = self.phi(tokens)                      # (B, N+1, obj_emb_dim)

        # Permutation-invariant aggregation: max + sum
        agg_max, _ = emb.max(dim=1)                 # (B, obj_emb_dim)
        agg_sum    = emb.sum(dim=1)                 # (B, obj_emb_dim)
        return torch.cat([agg_max, agg_sum], dim=-1)  # (B, 2*obj_emb_dim)

    def _encode_push(self, pushes: dict) -> torch.Tensor:
        """Encode push geometry → (B, push_emb_dim)."""
        start = pushes['push_start_xy'].float()   # (B, 2)
        end   = pushes['push_end_xy'].float()     # (B, 2)
        z     = pushes['push_z'].float()          # (B,) or (B,1)
        if z.dim() == 1:
            z = z.unsqueeze(-1)

        direction = end - start
        norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction_unit = direction / norm           # unit vector (B, 2)

        push_feat = torch.cat([start, direction_unit, z], dim=-1)  # (B, 5)
        return self.psi(push_feat)


# ---------------------------------------------------------------------------
# PPNFlat — flat MLP matching AlphaZero SolverNet style
# ---------------------------------------------------------------------------

def _push_features(pushes: dict) -> torch.Tensor:
    """Shared push feature encoder: (B, 5) — start_xy, dir_unit, z."""
    start = pushes['push_start_xy'].float()
    end   = pushes['push_end_xy'].float()
    z     = pushes['push_z'].float()
    if z.dim() == 1:
        z = z.unsqueeze(-1)
    direction = end - start
    norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.cat([start, direction / norm, z], dim=-1)  # (B, 5)


class PPNFlat(nn.Module):
    """
    Flat-MLP Push Prediction Network.

    Input: section-contiguous scene encoding (matching AlphaZero's
    ``encode_solver_state``, minus grid one-hots) concatenated with push
    features.  Total input dim = 7*(1+N) + 5.

    Parameters
    ----------
    n_obstacles : int
    hidden : int
        Width of both hidden layers (default 128, matching AlphaZero's trunk).
    """

    PUSH_DIM = 5  # start_xy(2) + dir_unit(2) + z(1)

    def __init__(self, n_obstacles: int, hidden: int = 128):
        super().__init__()
        self.n_obstacles = n_obstacles
        in_dim = 7 * (1 + n_obstacles) + self.PUSH_DIM
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------

    def forward(self, states: dict, pushes: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        states : dict
            'target_pos'    : (B, 3)
            'target_quat'   : (B, 4)
            'obstacle_pos'  : (B, N, 3)
            'obstacle_quat' : (B, N, 4)
        pushes : dict
            'push_start_xy' : (B, 2)
            'push_end_xy'   : (B, 2)
            'push_z'        : (B,)

        Returns
        -------
        q : (B,)
        """
        scene = self._encode_scene(states)          # (B, 7*(1+N))
        push  = _push_features(pushes)              # (B, 5)
        x = torch.cat([scene, push], dim=-1)
        return self.head(self.trunk(x)).squeeze(-1)

    def forward_single(self, state: dict, push: dict) -> torch.Tensor:
        """Unbatched forward for use inside tree search → scalar."""
        batched_state = {
            'target_pos':    state['target_pos'].unsqueeze(0),
            'target_quat':   state['target_quat'].unsqueeze(0),
            'obstacle_pos':  state['obstacle_pos'].unsqueeze(0),
            'obstacle_quat': state['obstacle_quat'].unsqueeze(0),
        }
        batched_push = {
            'push_start_xy': push['push_start_xy'].unsqueeze(0),
            'push_end_xy':   push['push_end_xy'].unsqueeze(0),
            'push_z':        torch.as_tensor(push['push_z']).float().unsqueeze(0),
        }
        return self.forward(batched_state, batched_push).squeeze(0)

    # ------------------------------------------------------------------

    def _encode_scene(self, states: dict) -> torch.Tensor:
        """Section-contiguous: target_xyz + obs_xyz + target_quat + obs_quat → (B, 7*(1+N))."""
        B = states['target_pos'].shape[0]
        N = self.n_obstacles

        t_pos  = states['target_pos'].float()[:, :3]           # (B, 3)
        t_quat = _canonical_quat(states['target_quat'].float()) # (B, 4)

        obs_pos  = states['obstacle_pos'].float()[:, :N, :3]   # (B, N, 3)
        obs_quat = states['obstacle_quat'].float()[:, :N, :]   # (B, N, 4)

        # Pad if scene has fewer obstacles than expected
        actual_n = obs_pos.shape[1]
        if actual_n < N:
            pad_n = N - actual_n
            obs_pos  = F.pad(obs_pos,  (0, 0, 0, pad_n))
            obs_quat = F.pad(obs_quat, (0, 0, 0, pad_n))

        # Canonicalize obstacle quats: reshape (B, N, 4) → (B*N, 4) → back
        obs_quat_flat = _canonical_quat(obs_quat.reshape(B * N, 4))
        obs_quat = obs_quat_flat.reshape(B, N, 4)

        return torch.cat([
            t_pos,                       # (B, 3)
            obs_pos.reshape(B, N * 3),   # (B, 3N)
            t_quat,                      # (B, 4)
            obs_quat.reshape(B, N * 4),  # (B, 4N)
        ], dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ppn(arch: str, n_obstacles: int, **kwargs) -> 'PPN | PPNFlat':
    """
    Construct a PPN by architecture name.

    Parameters
    ----------
    arch : str
        ``'deepsets'`` → :class:`PPN`, ``'mlp'`` → :class:`PPNFlat`.
    n_obstacles : int
    **kwargs
        Passed to the chosen class constructor.
        For ``'deepsets'``: ``obj_emb_dim``, ``push_emb_dim``, ``agg_hidden``.
        For ``'mlp'``: ``hidden``.
    """
    if arch == 'deepsets':
        return PPN(n_obstacles=n_obstacles, **kwargs)
    elif arch == 'mlp':
        return PPNFlat(n_obstacles=n_obstacles, **kwargs)
    else:
        raise ValueError(f"Unknown PPN arch '{arch}'. Choose 'deepsets' or 'mlp'.")
