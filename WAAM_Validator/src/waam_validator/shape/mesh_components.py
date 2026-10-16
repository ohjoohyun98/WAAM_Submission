"""Small mesh connectivity helpers without optional graph libraries."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import trimesh


def _edge_direction(face: npt.NDArray[np.int64], left: int, right: int) -> int:
    for index in range(3):
        start = int(face[index])
        stop = int(face[(index + 1) % 3])
        if start == left and stop == right:
            return 1
        if start == right and stop == left:
            return -1
    raise ValueError("Adjacent faces do not contain their reported shared edge.")


def _find_with_parity(
    node: int,
    parent: list[int],
    parity: bytearray,
) -> tuple[int, int]:
    """Return the union-find root and parity with iterative path compression."""
    root = node
    total_parity = 0
    while parent[root] != root:
        total_parity ^= parity[root]
        root = parent[root]

    current = node
    remaining_parity = total_parity
    while parent[current] != current:
        next_node = parent[current]
        edge_parity = parity[current]
        parent[current] = root
        parity[current] = remaining_parity
        remaining_parity ^= edge_parity
        current = next_node
    return root, total_parity


def repair_normals_and_count_bodies(mesh: trimesh.Trimesh) -> int:
    """Orient faces, fix closed-body inversion, and return the body count.

    Trimesh delegates these operations to optional NetworkX or SciPy graph engines.
    The Validator only needs face connectivity, so a parity union-find keeps the
    installation small and makes the STL path work in a clean environment.
    """
    face_count = len(mesh.faces)
    if face_count == 0:
        return 0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    shared_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    # Parent/rank/parity are scalar traversal state, so Python containers avoid
    # repeated NumPy scalar conversion. Bulk geometry arrays remain NumPy-backed.
    parent = list(range(face_count))
    rank = bytearray(face_count)
    parity = bytearray(face_count)

    for pair, edge in zip(adjacency, shared_edges, strict=True):
        left_face = int(pair[0])
        right_face = int(pair[1])
        edge_left = int(edge[0])
        edge_right = int(edge[1])
        required_parity = int(
            _edge_direction(faces[left_face], edge_left, edge_right)
            == _edge_direction(faces[right_face], edge_left, edge_right)
        )
        left_root, left_parity = _find_with_parity(left_face, parent, parity)
        right_root, right_parity = _find_with_parity(right_face, parent, parity)
        if left_root == right_root:
            if left_parity ^ right_parity != required_parity:
                raise ValueError("Mesh face winding constraints are inconsistent.")
            continue
        relation = left_parity ^ right_parity ^ required_parity
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
            parity[left_root] = relation
        else:
            parent[right_root] = left_root
            parity[right_root] = relation
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1

    roots = np.empty(face_count, dtype=np.int64)
    flips = np.zeros(face_count, dtype=bool)
    for face_index in range(face_count):
        root, flip = _find_with_parity(face_index, parent, parity)
        roots[face_index] = root
        flips[face_index] = bool(flip)
    if np.any(flips):
        repaired_faces = faces.copy()
        repaired_faces[flips] = np.fliplr(repaired_faces[flips])
        mesh.faces = repaired_faces

    unique_roots = np.unique(roots)
    if mesh.is_watertight:
        signed_volumes = np.zeros(face_count, dtype=np.float64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        repaired_faces = np.asarray(mesh.faces, dtype=np.int64)
        chunk_size = 100_000
        for start in range(0, face_count, chunk_size):
            stop = min(face_count, start + chunk_size)
            triangles = vertices[repaired_faces[start:stop]]
            face_volumes = (
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
                / 6.0
            )
            np.add.at(signed_volumes, roots[start:stop], face_volumes)
        inverted_faces = (signed_volumes < 0.0)[roots]
        if np.any(inverted_faces):
            repaired_faces = repaired_faces.copy()
            repaired_faces[inverted_faces] = np.fliplr(repaired_faces[inverted_faces])
            mesh.faces = repaired_faces
    return int(len(unique_roots))
