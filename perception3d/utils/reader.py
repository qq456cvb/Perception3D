import numpy as np
import warnings


def sample_vertex_from_mesh(vertex, facet, colors=None, rnd_idxs=None, u=None, v=None, num_samples=2048):
    scale = np.max(np.abs(vertex))
    vertex = vertex / scale
    triangles = np.take(vertex, facet, axis=0)
    if colors is not None:
        trianlges_color = np.take(colors, facet, axis=0)
    vx, vy, vz = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    if colors is not None:
        cx, cy, cz = trianlges_color[:, 0, :], trianlges_color[:, 1, :], trianlges_color[:, 2, :]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(vy - vx, vz - vx), axis=1)
    if np.sum(triangle_areas) < 1e-7:
        warnings.warn('Warning: not a good triangle mesh')
        idx = np.random.randint(0, vertex.shape[0], (num_samples,))
        return vertex[idx] * scale
    probs = triangle_areas / np.sum(triangle_areas)

    if rnd_idxs is None:
        rnd_idxs = np.random.choice(np.arange(probs.shape[0]), size=num_samples, p=probs)
    vx, vy, vz = vx[rnd_idxs], vy[rnd_idxs], vz[rnd_idxs]
    if colors is not None:
        cx, cy, cz = cx[rnd_idxs], cy[rnd_idxs], cz[rnd_idxs]
    if u is None:
        u = np.random.rand(vx.shape[0], 1).astype(vertex.dtype)
    if v is None:
        v = np.random.rand(vx.shape[0], 1).astype(vertex.dtype)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)
    pts = (vx * u + vy * v + vz * w) * scale
    if colors is not None:
        c = (cx * u + cy * v + cz * w)
        return pts, c, rnd_idxs, u, v
    else:
        return pts, rnd_idxs, u, v


def read_off(file):
    line = file.readline().strip()
    if 'OFF' != line:
        if line[:3] != 'OFF':
            raise('Not a valid OFF header')
        else:
            info = line[3:]
    else:
        info = file.readline().strip()
    n_verts, n_faces, _ = tuple([int(s) for s in info.split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
    return np.asarray(verts), np.asarray(faces)