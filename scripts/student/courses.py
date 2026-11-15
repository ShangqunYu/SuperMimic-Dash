"""Obstacle courses for training the depth student at speed (dagger.py --courses), and the fast eval scenes.

A course is 28 m along +x (4 cm heightfield, 4 m wide): flat start, then 1-4 features in a row with 1-3 m of flat
between them, the level carried from one to the next. Features: stairs up / down (riser 8-19 cm, tread 24-36 cm,
3-8 steps), a box in the way (12-32 cm high, 20-50 cm deep, 0.6-2 m wide), a platform up / down (8-32 cm),
a rough patch (2-4 m of random blocks 0.25-0.5 m a side, 0-4..20 cm high), a slope up / down (5-15 deg over
1.5-3 m) and a gap (10-40 cm wide, 1 m deep). A share of courses is flat ground. The joystick changes every 2-4 s:
on courses vx 0.3-2 m/s (half of the segments 1-2 m/s), a fifth with vy up to 0.3 m/s, half aiming up to 20 deg off
the course; on flat ground augment.py's flat schedule with vx up to 2 m/s. In probes the generator + v6 teacher
crossed stairs, boxes, platforms, blocks up to 20 cm, slopes up to 15 deg and gaps up to 40 cm at 1.3-2 m/s.
"""

from __future__ import annotations

import numpy as np

from supermimic.terrain.terrains import Terrain

DX, X0, X1, Y0, Y1 = 0.04, -1.0, 27.0, -2.0, 2.0
SEG = (2.0, 4.0)
FEATURES = {"stairs_up": 0.17, "stairs_down": 0.13, "box": 0.14, "platform_up": 0.1, "platform_down": 0.08, "rough": 0.16, "slope_up": 0.08,
            "slope_down": 0.06, "gap": 0.08}
RISE = (0.08, 0.19)  # m, stair riser
TALL_RISE, TALL_SHARE = (0.15, 0.19), 0.0  # a share of the stairs drawn from TALL_RISE instead (rl_finetune.py --tall-stairs)
GAP = (0.1, 0.4)  # m, gap width (rl_finetune.py --gap-max)


def grid(x0=X0, x1=X1, y0=Y0, y1=Y1, dx=DX):
    x = x0 + dx * np.arange(round((x1 - x0) / dx) + 1)
    y = y0 + dx * np.arange(round((y1 - y0) / dx) + 1)
    return np.meshgrid(x, y, indexing="ij")


def _feature(rng, kind: str, X, Y, x0: float, level: float):
    """-> heightfield increment over the feature's extent (added where X >= x0), its end x, the level after it."""
    if kind in ("stairs_up", "stairs_down"):
        r = rng.uniform(*TALL_RISE) if TALL_SHARE and rng.random() < TALL_SHARE else rng.uniform(*RISE)  # no extra draw at 0
        t, n = rng.uniform(0.24, 0.36), int(rng.integers(3, 9))
        k = np.clip(np.floor((X - x0) / t) + 1, 0, n)
        d = k * r if kind == "stairs_up" else -k * r
        return d, x0 + n * t, level + (n * r if kind == "stairs_up" else -n * r)
    if kind == "box":
        h, dd, w = rng.uniform(0.12, 0.32), rng.uniform(0.2, 0.5), rng.uniform(0.6, 2.0)
        return np.where((X >= x0) & (X < x0 + dd) & (np.abs(Y) < w / 2), h, 0.0), x0 + dd, level
    if kind in ("platform_up", "platform_down"):
        h = rng.uniform(0.08, 0.32) * (1 if kind == "platform_up" else -1)
        return np.where(X >= x0, h, 0.0), x0, level + h
    if kind == "rough":
        length, cell, hmax = rng.uniform(2.0, 4.0), rng.uniform(0.25, 0.5), rng.uniform(0.04, 0.2)
        ix = np.floor((X - x0) / cell).astype(int).clip(0)
        iy = np.floor((Y - Y0) / cell).astype(int).clip(0)
        blocks = rng.uniform(0, hmax, (ix.max() + 1, iy.max() + 1))
        return np.where((X >= x0) & (X < x0 + length), blocks[ix, iy], 0.0), x0 + length, level
    if kind in ("slope_up", "slope_down"):
        g, length = np.tan(np.radians(rng.uniform(5.0, 15.0))), rng.uniform(1.5, 3.0)
        d = np.clip(X - x0, 0, length) * g * (1 if kind == "slope_up" else -1)
        return d, x0 + length, level + length * g * (1 if kind == "slope_up" else -1)
    w = rng.uniform(*GAP)  # gap
    return np.where((X >= x0) & (X < x0 + w), -1.0, 0.0), x0 + w, level


def sample_course(rng: np.random.Generator, flat_share: float = 0.2):
    """-> (family, params, terrain, start ground, end ground, x to pass (None: flat), flat?) as augment.sample_scene's
    leading fields."""
    X, Y = grid()
    if rng.random() < flat_share:
        return "flat", "flat", Terrain(np.zeros_like(X, dtype=np.float32), np.array([X0, Y0]), DX), 0.0, 0.0, None
    hf = np.zeros_like(X)
    x, level = rng.uniform(1.0, 2.0), 0.0
    kinds = []
    for _ in range(int(rng.integers(1, 5))):
        names = list(FEATURES)
        p = np.array([FEATURES[k] for k in names])
        if level < -1.0:  # keep the level within about -1.5..2 m
            p[[names.index(k) for k in ("stairs_down", "platform_down", "slope_down")]] = 0
        if level > 1.5:
            p[[names.index(k) for k in ("stairs_up", "platform_up", "slope_up")]] = 0
        kind = rng.choice(names, p=p / p.sum())
        d, end, level = _feature(rng, kind, X, Y, x, level)
        hf += d
        kinds.append(kind)
        x = end + rng.uniform(1.0, 3.0)
        if x > X1 - 6.0:
            break
    past = end + 0.3
    hf = hf.astype(np.float32)
    ter = Terrain(hf, np.array([X0, Y0]), DX)
    return "course", "+".join(kinds), ter, 0.0, float(level), past


def course_schedule(rng: np.random.Generator, flat: bool, steps: int, side: float = 0.25, vmax: float = 2.0) -> np.ndarray:
    """(steps, 4) vx, vy, wz (0: steer to the target heading), target heading; a new joystick every 2-4 s."""
    out = np.zeros((steps, 4), np.float32)
    t, tgt = 0, 0.0
    while t < steps:
        n = min(int(rng.uniform(*SEG) * 50), steps - t)
        vx = vy = wz = 0.0
        if flat:
            r = rng.random()
            if 0.15 <= r < 0.15 + side:
                vx, vy = rng.uniform(-0.1, 0.1), float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.6))
            elif r >= 0.15 + side:
                vx, vy = rng.uniform(-0.4, vmax), (rng.uniform(-0.5, 0.5) if rng.random() < 0.6 else 0.0)
                if (r - 0.15 - side) / (0.85 - side) < 0.3 / 0.85:
                    wz = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.8))
                elif rng.random() < 0.5:
                    tgt += rng.uniform(-np.pi / 2, np.pi / 2)
        else:
            vx = rng.uniform(1.0, vmax) if rng.random() < 0.5 else rng.uniform(0.3, 1.0)
            vy = rng.uniform(-0.3, 0.3) if t > 0 and rng.random() < 0.2 else 0.0
            tgt = rng.uniform(-0.35, 0.35) if t > 0 and rng.random() < 0.5 else 0.0
        out[t : t + n] = vx, vy, wz, tgt
        tgt += wz * n * 0.02
        t += n
    return out


def rough_scenes(speed: float):
    """Eval scenes on terrain the generator never saw (rollout_generator's grid): random blocks between x 1 and 5 m,
    slopes, gaps; joystick vx `speed`."""
    from rollout_generator import Scene, grid as eval_grid

    X, Y = eval_grid()
    rng = np.random.default_rng(0)
    out = []
    zone = (X >= 1.0) & (X < 5.0)
    for hmax, cell in ((0.06, 0.3), (0.12, 0.3), (0.20, 0.5)):
        ix, iy = np.floor((X - 1.0) / cell).astype(int).clip(0), np.floor((Y + 2.0) / cell).astype(int).clip(0)
        blocks = rng.uniform(0, hmax, (ix.max() + 1, iy.max() + 1))
        out.append(Scene(f"rough blocks 0-{hmax * 100:.0f} cm", np.where(zone, blocks[ix, iy], 0.0), (speed, 0, 0), top=0.0, past=5.5))
    for deg in (10, 15):
        g = np.tan(np.radians(deg))
        up = np.clip(X - 1.5, 0, 2.5) * g
        out.append(Scene(f"slope up {deg} deg", up, (speed, 0, 0), top=2.5 * g, past=4.5))
        out.append(Scene(f"slope down {deg} deg", 2.5 * g - up, (speed, 0, 0), top=0.0, past=4.5))
    for w in (0.2, 0.4):
        out.append(Scene(f"gap {w * 100:.0f} cm", np.where((X >= 2.0) & (X < 2.0 + w), -1.0, 0.0), (speed, 0, 0), top=0.0, past=3.0))
    return out


def fast_scenes(speed: float = 1.5):
    """rollout_generator's terrain scenes and rough_scenes, all at joystick vx `speed`."""
    import dataclasses

    from rollout_generator import scenes

    return [dataclasses.replace(s, cmd=(speed, 0.0, 0.0)) for s in scenes() if s.past is not None] + rough_scenes(speed)


def fast_score(sc, terr, Q, tiles, scan, S: int) -> tuple[dict, int]:
    """Fast suite: success = past the scene's line, never tilted past 60 deg, standing at the end (root > 0.45 m
    above the ground). Gaps make rollout_generator.score's height rules fail by construction, so it is not used.
    -> per scene {success, crossing speed m/s (x 0.5 m -> the line, mean of those that cross)}, successes."""
    x = Q[..., 0]
    qx, qy = Q[..., 4], Q[..., 5]
    upright = (1 - 2 * (qx**2 + qy**2)).amin(1) > 0.5
    ground = scan.lookup(Q[:, -1:, 0], Q[:, -1:, 1], tiles)[:, 0]
    standing = Q[:, -1, 2] - ground > 0.45
    res, total = {}, 0
    for k, s in enumerate(sc):
        sl = slice(k * S, (k + 1) * S)
        xs = x[sl]
        ok = (xs[:, -1] > s.past) & upright[sl] & standing[sl]
        i0 = (xs > 0.5).float().argmax(1)
        i1 = (xs > s.past).float().argmax(1)
        crossed = (xs > s.past).any(1) & (i1 > i0)
        v = ((s.past - 0.5) / ((i1 - i0).clamp_min(1) * 0.02))[crossed]
        res[s.name] = {"success": f"{int(ok.sum())}/{S}", "crossing_m_s": round(float(v.mean()), 2) if len(v) else None}
        total += int(ok.sum())
    return res, total
