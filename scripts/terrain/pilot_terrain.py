"""Terrain pilot: recover box/stair geometry from retargeted BONES-SEED terrain clips, score it, render it.

BONES-SEED has no terrain geometry, only prop labels ("50cm box", "stairs"). The geometry is
recovered from where Dash's feet rest:
  1. stance: a foot-capsule end (heel or toe) that is nearly still for >= ``min_stance`` s;
  2. its sole height (capsule end z - radius) is a foothold; footholds are clustered into flat
     height levels (gaps > ``level_gap``), each snapped to its level's median;
  3. footholds are stamped into a 1 cm height grid and the rest is filled from the nearest
     foothold (so a step edge lies halfway between the last lower and first upper foothold).
Scores (Dash scale, metres): level heights, stance flatness, and how far the foot and shin
capsules go into the recovered terrain outside stance (penetration).

  uv run python scripts/terrain/pilot_terrain.py --raw data/terrain_pilot/raw --out logs/terrain_pilot
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from supermimic.robots import get_robot

FPS_IN = 120
STEP = 2  # analyse/render at 60 fps
CELL = 0.01
HUMAN_BONES = ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
               "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
               "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd")


def to_qpos(q: np.ndarray) -> np.ndarray:
    """mocap_retargeting q (pos, quat xyzw, joints) -> MuJoCo qpos (pos, quat wxyz, joints)."""
    out = q.copy()
    out[:, 3] = q[:, 6]
    out[:, 4:7] = q[:, 3:6]
    return out


def capsule_points(m, d, g, n=5):
    """World points along the bottom of capsule geom g: (n, 3) axis points and its radius."""
    c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    r, h = m.geom_size[g, 0], m.geom_size[g, 1]
    t = np.linspace(-h, h, n)
    return c + t[:, None] * R[:, 2], r


def track(m, qpos):
    """Per frame: foot capsule ends (T, 2 feet, 2 ends, 3), shin points (T, 2, 5, 3), radii."""
    d = mujoco.MjData(m)
    feet = [m.geom(f"{s}_foot_collision").id for s in "lr"]
    shins = [m.geom(f"{s}_lower_leg_collision").id for s in "lr"]
    fe, sp = [], []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        fe.append([capsule_points(m, d, g, 2)[0] for g in feet])
        sp.append([capsule_points(m, d, g, 5)[0] for g in shins])
    return np.array(fe), np.array(sp), m.geom_size[feet[0], 0], m.geom_size[shins[0], 0]


def stance_mask(p: np.ndarray, dt: float, vmax: float, min_len: int) -> np.ndarray:
    """p (T, 3) -> bool (T,): speed below vmax for at least min_len consecutive frames."""
    v = np.linalg.norm(np.gradient(p, dt, axis=0), axis=-1)
    still = v < vmax
    lab, n = ndimage.label(still)
    keep = np.zeros_like(still)
    for i in range(1, n + 1):
        idx = lab == i
        if idx.sum() >= min_len:
            keep |= idx
    return keep


def cluster_levels(h: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D clustering of heights: sorted values split where consecutive gap > ``gap``."""
    order = np.argsort(h)
    hs = h[order]
    cut = np.flatnonzero(np.diff(hs) > gap) + 1
    lab_sorted = np.zeros(len(h), dtype=int)
    for c in cut:
        lab_sorted[c:] += 1
    lab = np.empty_like(lab_sorted)
    lab[order] = lab_sorted
    levels = np.array([np.median(h[lab == k]) for k in range(lab.max() + 1)])
    return lab, levels


def recover(fe, foot_r, dt, vmax=0.12, min_stance=0.1, level_gap=0.04, stamp_r=0.025):
    T = fe.shape[0]
    for v, ms in ((vmax, min_stance), (0.25, 0.05), (0.4, 0.03)):  # relax for jogging (short, moving stance)
        pts, hs = [], []
        stance = np.zeros((T, 2, 2), dtype=bool)
        for f in range(2):
            for e in range(2):
                s = stance_mask(fe[:, f, e], dt, v, max(2, int(round(ms / dt))))
                stance[:, f, e] = s
                pts.append(fe[s, f, e, :2])
                hs.append(fe[s, f, e, 2] - foot_r)
        pts, hs = np.concatenate(pts), np.concatenate(hs)
        if len(hs) >= 20:
            break
    lab, levels = cluster_levels(hs, level_gap)
    # Drop levels with very few footholds (a single brief toe-rest mid-air etc.).
    counts = np.bincount(lab, minlength=len(levels))
    good = counts >= max(6, int(0.02 * len(hs)))
    keep = good[lab]
    pts, hs, lab = pts[keep], hs[keep], lab[keep]
    remap = -np.ones(len(levels), dtype=int)
    remap[good] = np.arange(good.sum())
    lab, levels_kept = remap[lab], levels[good]
    flat_std = np.array([hs[lab == k].std() for k in range(len(levels_kept))])
    return pts, hs, lab, levels_kept, flat_std, stance


def heightfield(pts, lab, levels, extent, stamp_r):
    (x0, x1), (y0, y1) = extent
    nx, ny = int(np.ceil((x1 - x0) / CELL)) + 1, int(np.ceil((y1 - y0) / CELL)) + 1
    lvl = -np.ones((ny, nx), dtype=int)
    ix = np.clip(((pts[:, 0] - x0) / CELL).round().astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / CELL).round().astype(int), 0, ny - 1)
    k = int(np.ceil(stamp_r / CELL))
    # Higher levels stamped last win where stamps overlap (a toe resting on a step edge).
    for L in np.argsort(levels):
        sel = lab == L
        grid = np.zeros_like(lvl, dtype=bool)
        grid[iy[sel], ix[sel]] = True
        grid = ndimage.binary_dilation(grid, iterations=k)
        lvl[grid] = L
    _, (iy_n, ix_n) = ndimage.distance_transform_edt(lvl < 0, return_indices=True)
    lvl = lvl[iy_n, ix_n]
    return levels[lvl], (x0, y0)


def profile_heightfield(pts, lab, levels, fe, sp, foot_r, shin_r, stance, extent, stairs: bool):
    """Straight-edged terrain: a piecewise-constant height profile h(s) along the walking direction.

    Levels are ordered along s by their footholds; each edge is searched (1 cm) inside the gap
    between the neighbouring levels' footholds for the least swing-foot + shin penetration. Stairs
    are extended past the first/last stepped level with the mean rise and run (loop clips start
    and end mid-staircase). Returns H, origin, d (unit direction), edges, heights, or None when the
    footholds do not lie along a line (e.g. walking along an edge with turns)."""
    c = pts.mean(0)
    L = len(levels)
    if L < 2:
        return None
    # Direction: from the lowest level's footholds to the highest level's.
    d = pts[lab == np.argmax(levels)].mean(0) - pts[lab == np.argmin(levels)].mean(0)
    if np.linalg.norm(d) < 0.05:
        return None
    d = d / np.linalg.norm(d)
    s_pts = (pts - c) @ d
    mean_s = np.array([s_pts[lab == k].mean() for k in range(L)])
    order = np.argsort(mean_s)
    if mean_s[order[-1]] < mean_s[order[0]] + 0.05 and L > 1:
        return None
    # Monotonic height along s is required for this model (boxes, straight stairs).
    hs_ord = levels[order]
    if L > 1 and not (np.all(np.diff(hs_ord) > 0) or np.all(np.diff(hs_ord) < 0)):
        return None
    lo = np.array([s_pts[lab == k].min() for k in order])
    hi = np.array([s_pts[lab == k].max() for k in order])

    # Cost of an edge layout: any penetration (foot ends, stance or not, and shin points) is
    # expensive. A standing foot must rest on at least one end (heel or toe within 2 cm of the
    # surface); the other end may overhang (Dash's 18 cm foot is as long as a scaled 19 cm tread).
    fs = (fe[..., :2] - c) @ d  # (T,2,2)
    fz = fe[..., 2] - foot_r
    ss, sz = (sp[..., :2] - c) @ d, sp[..., 2] - shin_r
    standing = stance.any(-1)  # (T,2)

    def cost(edges, heights):
        hf = heights[np.searchsorted(edges, fs)]
        hsn = heights[np.searchsorted(edges, ss)]
        pen = np.clip(hf - fz, 0, None).sum() + np.clip(hsn - sz, 0, None).sum()
        unsupported = np.clip(np.abs(fz - hf).min(-1) - 0.02, 0, None)[standing].sum()
        return pen + unsupported

    heights = hs_ord.copy()
    edges = np.array([(hi[i] + lo[i + 1]) / 2 for i in range(L - 1)])
    for _ in range(3):  # coordinate descent; each edge stays between its neighbours
        for i in range(L - 1):
            a = max(lo[i], edges[i - 1] + 0.02) if i > 0 else lo[i]
            b = min(hi[i + 1], edges[i + 1] - 0.02) if i < L - 2 else hi[i + 1]
            best = (cost(edges, heights), edges[i])
            for e in np.arange(a, b + 1e-9, 0.01):
                edges[i] = e
                p = cost(edges, heights)
                if p < best[0]:
                    best = (p, e)
            edges[i] = best[1]
    if stairs and L >= 3:
        rise = np.median(np.diff(heights))
        run = np.median(np.diff(edges)) if len(edges) >= 2 else 0.25
        s_all = np.concatenate([fs.ravel(), ss.ravel()])
        while edges[0] > s_all.min() - 0.3 and len(edges) < 60:
            edges = np.concatenate([[edges[0] - run], edges])
            heights = np.concatenate([[heights[0] - rise], heights])
        while edges[-1] < s_all.max() + 0.3 and len(edges) < 60:
            edges = np.concatenate([edges, [edges[-1] + run]])
            heights = np.concatenate([heights, [heights[-1] + rise]])
        heights = np.maximum(heights, min(0.0, levels.min()))  # never below the floor
    (x0, x1), (y0, y1) = extent
    nx, ny = int(np.ceil((x1 - x0) / CELL)) + 1, int(np.ceil((y1 - y0) / CELL)) + 1
    X, Y = np.meshgrid(x0 + CELL * np.arange(nx), y0 + CELL * np.arange(ny))
    S = (X - c[0]) * d[0] + (Y - c[1]) * d[1]
    H = heights[np.searchsorted(edges, S)]
    return H, (x0, y0), d, edges, heights


def sample_h(H, origin, xy):
    x0, y0 = origin
    ix = np.clip(((xy[..., 0] - x0) / CELL).round().astype(int), 0, H.shape[1] - 1)
    iy = np.clip(((xy[..., 1] - y0) / CELL).round().astype(int), 0, H.shape[0] - 1)
    return H[iy, ix]


def score(H, origin, fe, sp, foot_r, shin_r, stance):
    foot_pen = sample_h(H, origin, fe[..., :2]) - (fe[..., 2] - foot_r)  # (T,2,2)
    shin_pen = sample_h(H, origin, sp[..., :2]) - (sp[..., 2] - shin_r)  # (T,2,5)
    swing = ~stance
    fp = np.where(swing, foot_pen, -np.inf).max(axis=(1, 2))
    support = np.abs(foot_pen).min(-1)  # (T,2): closer end of each foot to the surface
    support = support[stance.any(-1)]
    return {
        "foot_pen_max": float(np.max(fp)),
        "foot_pen_frac_1cm": float(np.mean(fp > 0.01)),
        "shin_pen_max": float(shin_pen.max()),
        "shin_pen_frac_1cm": float(np.mean(shin_pen.max(axis=(1, 2)) > 0.01)),
        "support_gap_p95": float(np.percentile(support, 95)),
    }


def build_scene(robot_xml, H, origin, width, height, has_human):
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.quality.shadowsize = 4096
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.84, 0.86], rgb2=[0.74, 0.76, 0.79], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    wb = spec.worldbody
    base = min(float(H.min()), 0.0) - 0.002
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], pos=[0, 0, base - 0.01], material="grid", contype=0, conaffinity=0)
    ny, nx = H.shape
    top = float(H.max() - base)
    hf = spec.add_hfield(name="terrain", nrow=ny, ncol=nx,
                         size=[(nx - 1) * CELL / 2, (ny - 1) * CELL / 2, max(top, 1e-3), 0.02])
    hf.userdata = ((H - base) / max(top, 1e-3)).astype(np.float32).ravel().tolist()
    cx, cy = origin[0] + (nx - 1) * CELL / 2, origin[1] + (ny - 1) * CELL / 2
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", pos=[cx, cy, base], rgba=[0.55, 0.62, 0.72, 1], contype=0, conaffinity=0)
    wb.add_light(pos=[0, 0, 30], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6] * 3, specular=[0.1] * 3, castshadow=True)
    wb.add_light(pos=[10, -10, 15], dir=[-1, 1, -1.5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.3] * 3, castshadow=False)
    return spec.compile()


def add_line(scene, a, b, rgba, width=0.008):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
    scene.ngeom += 1


def render_clip(name, label, qpos, mocap, bones, H, origin, travel, stats, out, width=640, height=480, fps=60):
    m = build_scene(get_robot("dash").xml_path(), H, origin, width, height, mocap is not None)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 2.2, -12.0
    cam.azimuth = float(np.degrees(np.arctan2(travel[1], travel[0]))) - 90.0  # side view of the travel direction
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(out)], stdin=subprocess.PIPE)
    look = qpos[0, :3].copy()
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * qpos[t, :3] if t else qpos[t, :3].copy()
        cam.lookat[:] = [look[0], look[1], look[2] - 0.25]
        r.update_scene(d, cam)
        if mocap is not None:
            for a, b in bones:
                add_line(r.scene, mocap[t, a], mocap[t, b], (1.0, 0.45, 0.1, 0.55))
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        dr.text((8, 6), name.removeprefix("dash_"), fill=(10, 10, 10), font=font)
        dr.text((8, 26), label, fill=(20, 20, 130), font=font)
        dr.text((8, 46), stats, fill=(130, 20, 20), font=font)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()


def main(raw: Path = Path("data/terrain_pilot/raw"), meta: Path = Path("data/terrain_pilot/pilot_meta.csv"), out: Path = Path("logs/terrain_pilot"),
         render: bool = True, human: bool = True, only: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(get_robot("dash").xml_path()))
    info = {r["name"]: r for r in csv.DictReader(open(meta))}
    dt = STEP / FPS_IN
    rows = []
    for f in sorted(raw.glob(f"dash_*{only}*.npy" if only else "dash_*.npy")):
        name = f.stem
        d = np.load(f, allow_pickle=True).item()
        qpos = to_qpos(d["q"])[::STEP]
        fe, sp, foot_r, shin_r = track(m, qpos)
        pts, hs, lab, levels, flat_std, stance = recover(fe, foot_r, dt)
        lo = np.minimum(fe[..., :2].reshape(-1, 2).min(0), qpos[:, :2].min(0)) - 0.6
        hi = np.maximum(fe[..., :2].reshape(-1, 2).max(0), qpos[:, :2].max(0)) + 0.6
        clip = name.removeprefix("dash_")
        content = info.get(clip, {}).get("content", "")
        ext = ((lo[0], hi[0]), (lo[1], hi[1]))
        prof = profile_heightfield(pts, lab, levels, fe, sp, foot_r, shin_r, stance, ext, stairs="stair" in content)
        if prof is not None:
            H, origin, travel, _, _ = prof
            model = "profile"
        else:
            H, origin = heightfield(pts, lab, levels, ext, 0.025)
            travel = qpos[-1, :2] - qpos[0, :2]
            travel = travel / (np.linalg.norm(travel) + 1e-9) if np.linalg.norm(travel) > 0.2 else np.array([1.0, 0.0])
            model = "nearest"
        sc = score(H, origin, fe, sp, foot_r, shin_r, stance)
        steps = np.diff(np.sort(levels))
        row = {"name": name, "content": content, "model": model, "seconds": len(qpos) * dt, "levels": " ".join(f"{v:.3f}" for v in levels),
               "max_rise": float(steps.max()) if len(steps) else 0.0, "flat_std_max": float(flat_std.max()), **sc}
        rows.append(row)
        print(f"{clip:44s} {model[:4]} levels [{row['levels'][:40]}]  flat {row['flat_std_max'] * 100:.1f}cm  foot pen max {sc['foot_pen_max'] * 100:5.1f}cm "
              f"({sc['foot_pen_frac_1cm'] * 100:4.1f}% fr >1cm)  shin pen {sc['shin_pen_max'] * 100:5.1f}cm  support gap p95 {sc['support_gap_p95'] * 100:.1f}cm")
        if render:
            mocap, bones = None, []
            if human and "mocap" in d:
                jn = list(d["joint_names"])
                idx = {n: i for i, n in enumerate(jn)}
                bones = [(idx[a], idx[b]) for a, b in d["skeleton"] if a in idx and b in idx and a in HUMAN_BONES and b in HUMAN_BONES]
                mocap = d["mocap"][::STEP]
            label = f"recovered levels (m): {row['levels']}"
            stats = f"foot pen max {sc['foot_pen_max'] * 100:.1f} cm, shin pen max {sc['shin_pen_max'] * 100:.1f} cm"
            render_clip(name, label, qpos, mocap, bones, H, origin, travel, stats, out / "clips" / f"{clip}.mp4")
    with open(out / "pilot_scores.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "pilot_scores.csv")


if __name__ == "__main__":
    tyro.cli(main)
