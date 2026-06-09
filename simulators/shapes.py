"""
Tetris-like multi-cell shape definitions and STL mesh generation.

Cell patterns: (row, col) offsets in a 2D grid.
Row maps to local X, col maps to local Y in the physics scene.
All shapes are 1 cell tall in Z and lie flat on the floor.

Meshes are generated as closed (watertight) surfaces by including only the
outer faces of the union of cells — inner shared faces are omitted so that
Genesis's convex decomposition (VHACD) receives a clean manifold mesh.
"""

import struct

PIECES: dict[str, list[tuple[int, int]]] = {
    'cube': [(0, 0)],
    'I':    [(0, 0), (1, 0), (2, 0), (3, 0)],
    'O':    [(0, 0), (0, 1), (1, 0), (1, 1)],
    'T':    [(0, 0), (0, 1), (0, 2), (1, 1)],
    'S':    [(0, 1), (0, 2), (1, 0), (1, 1)],
    'Z':    [(0, 0), (0, 1), (1, 1), (1, 2)],
    'L':    [(0, 0), (1, 0), (2, 0), (2, 1)],
    'J':    [(0, 1), (1, 1), (2, 0), (2, 1)],
}

NON_CUBE_PIECES = [k for k in PIECES if k != 'cube']


def shape_extents(cells: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (n_rows, n_cols) bounding-box size in cells."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _outer_triangles(cells: list[tuple[int, int]], cell_size: float) -> list:
    """Return STL triangles (normal, v0, v1, v2) for the outer surface only."""
    cell_set = set(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    cx = (max(rows) + min(rows)) / 2.0 * cell_size
    cy = (max(cols) + min(cols)) / 2.0 * cell_size
    half = cell_size / 2.0

    tris = []
    for r, c in cells:
        x0 = r * cell_size - cx - half
        y0 = c * cell_size - cy - half
        x1, y1 = x0 + cell_size, y0 + cell_size
        z0, z1 = -half, half

        # Bottom and top are always exposed (shapes are one layer tall)
        tris += [
            ((0, 0, -1), (x0, y0, z0), (x1, y0, z0), (x1, y1, z0)),
            ((0, 0, -1), (x0, y0, z0), (x1, y1, z0), (x0, y1, z0)),
            ((0, 0, +1), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1)),
            ((0, 0, +1), (x0, y0, z1), (x1, y1, z1), (x1, y0, z1)),
        ]
        if (r, c - 1) not in cell_set:  # front (−Y)
            tris += [
                ((0, -1, 0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1)),
                ((0, -1, 0), (x0, y0, z0), (x1, y0, z1), (x1, y0, z0)),
            ]
        if (r, c + 1) not in cell_set:  # back (+Y)
            tris += [
                ((0, +1, 0), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1)),
                ((0, +1, 0), (x0, y1, z0), (x1, y1, z1), (x0, y1, z1)),
            ]
        if (r - 1, c) not in cell_set:  # left (−X)
            tris += [
                ((-1, 0, 0), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)),
                ((-1, 0, 0), (x0, y0, z0), (x0, y1, z1), (x0, y0, z1)),
            ]
        if (r + 1, c) not in cell_set:  # right (+X)
            tris += [
                ((+1, 0, 0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1)),
                ((+1, 0, 0), (x1, y0, z0), (x1, y1, z1), (x1, y1, z0)),
            ]
    return tris


def write_shape_stl(cells: list[tuple[int, int]], cell_size: float, path: str) -> None:
    """Write a binary STL for the given cell pattern, centered at the mesh origin."""
    tris = _outer_triangles(cells, cell_size)
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(tris)))
        for normal, v0, v1, v2 in tris:
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))
