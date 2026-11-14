"""Kinematic rollouts of the motion generator (no physics) on procedural terrains it never saw, driven by a
joystick command: stairs up / down at several risers, boxes, platforms, flat walking and turning. Where the scene
asks for no turning, the operator steers back to the start heading (``operator``), as a person would.

  python scripts/generator/rollout_generator.py ckpt out_dir [--samples 4] [--seconds 8] [--render] [--steps 20]

Every 0.2 s the generator is sampled from the last 10 frames of the rollout itself, the heightmap at the current
root and the command path (velocity springs from the current root velocity to the command, half-life 0.3 s),
and the first 10 frames of its plan are appended. Per scene: foot penetration / floating / sliding, how far
it got, the ground height reached, the lowest root height above the ground, success (no foot > 4 cm into the terrain (mocap on its own terrain: p90 2-3 cm),
root 0.35-0.4 to 1.1 m above the ground, past the obstacle on the right level, speed in range; foot sliding is
reported only: mocap flat walking itself reads ~0.5 m/s), and with --render one video per
scene (first sample) plus all scenes side by side in grid.mp4.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from supermimic.generator.data import CMD_OFFSETS, GEN_GRID, GRID_SHAPE, from_canonical, to_canonical
from supermimic.generator.metrics import FootMetrics
from supermimic.generator.model import Denoiser, Diffusion, GenCfg
from supermimic.kinematics.rotations import heading_yaw, quat_from_yaw, quat_mul
from supermimic.terrain.height_scan import HeightScan
from supermimic.terrain.terrains import Terrain

REPO = Path(__file__).resolve().parents[2]
DX = 0.02
X0, X1, Y0, Y1 = -2.0, 9.0, -2.0, 2.0
REPLAN = 10  # frames between plans
VEL_HALFLIFE = 0.3


@dataclass
class Scene:
    name: str
    hf: np.ndarray
    cmd: tuple[float, float, float]  # vx, vy (heading frame), wz
    top: float = 0.0  # ground height a success ends on
    past: float | None = None  # x the root must end beyond (None: the commanded speed is checked instead)
    min_above: float = 0.4  # lowest root height above the ground allowed


def grid():
    x = X0 + DX * np.arange(round((X1 - X0) / DX) + 1)
    y = Y0 + DX * np.arange(round((Y1 - Y0) / DX) + 1)
    return np.meshgrid(x, y, indexing="ij")


def scenes() -> list[Scene]:
    X, Y = grid()
    out = []
    flat = np.zeros_like(X)
    out.append(Scene("flat walk 0.6 m/s", flat, (0.6, 0.0, 0.0)))
    out.append(Scene("flat jog 1.5 m/s", flat, (1.5, 0.0, 0.0)))
    out.append(Scene("flat walk + turn left", flat, (0.5, 0.0, 0.6)))
    out.append(Scene("flat sidestep right", flat, (0.0, -0.4, 0.0)))
    for r in (0.10, 0.14, 0.18):
        hf = np.clip(np.floor((X - 1.2) / 0.28) + 1, 0, 6) * r
        out.append(Scene(f"stairs up {r * 100:.0f} cm x6", hf, (0.5, 0.0, 0.0), top=6 * r, past=3.0))
    for r in (0.10, 0.14):
        hf = (6 - np.clip(np.floor((X - 1.2) / 0.28) + 1, 0, 6)) * r
        out.append(Scene(f"stairs down {r * 100:.0f} cm x6", hf, (0.5, 0.0, 0.0), top=0.0, past=3.0))
    for h in (0.20, 0.30):
        hf = np.where((X >= 1.5) & (X < 1.9) & (np.abs(Y) < 0.6), h, 0.0)
        out.append(Scene(f"box {h * 100:.0f} cm in the way (40 cm deep)", hf, (0.8, 0.0, 0.0), top=0.0, past=2.3, min_above=0.35))
    for h in (0.30, 0.50):
        out.append(Scene(f"platform up {h * 100:.0f} cm", np.where(X >= 1.5, h, 0.0), (0.6, 0.0, 0.0), top=h, past=1.8))
        out.append(Scene(f"platform down {h * 100:.0f} cm", np.where(X < 1.5, h, 0.0), (0.6, 0.0, 0.0), top=0.0, past=1.8))
    return out


def load(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = GenCfg(**ck["cfg"])
    m = Denoiser(cfg).to(device)
    m.load_state_dict(ck["ema"])
    return m.eval(), Diffusion(cfg.T, device), ck


def standing_frame(lib: str, device: str) -> torch.Tensor:
    """(25,) [root pos, quat, joints] of a standing frame (first frame of a walk start), root at x = y = 0, yaw 0."""
    pack = torch.load(lib, map_location="cpu", weights_only=False, mmap=True)
    i = next(k for k, n in enumerate(pack["names"]) if n.startswith("dash_walk_forward_start_"))
    f = pack["ref"][int(pack["starts"][i])].float().to(device)
    q = quat_mul(quat_from_yaw(-heading_yaw(f[3:7])), f[3:7])
    return torch.cat([torch.tensor([0.0, 0.0], device=device), f[2:3], q, f[7:25]])


HOLD_GAIN, HOLD_MAX = 2.0, 1.0  # heading-holding operator: wz = gain x heading error (rad/s), capped


def operator(traj: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """The joystick a person would give: the scene's command, but where it asks for no turning the yaw rate steers
    back to the start heading (yaw 0), as an operator corrects drift. traj (B, T, 25), cmd (B, 3) -> (B, 3)."""
    out = cmd.clone()
    hold = cmd[:, 2] == 0
    yaw = heading_yaw(traj[:, -1, 3:7])
    out[:, 2] = torch.where(hold, (-HOLD_GAIN * torch.atan2(torch.sin(yaw), torch.cos(yaw))).clamp(-HOLD_MAX, HOLD_MAX), cmd[:, 2])
    return out


def command_path(traj: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """traj (B, T, 25) rollout so far, cmd (B, 3) -> (B, 12) command features in the current frame: the root
    velocity springs from its recent value to the command (half-life VEL_HALFLIFE), integrated forward."""
    cur, prev = traj[:, -1], traj[:, -REPLAN - 1]
    yaw0, yawp = heading_yaw(cur[:, 3:7]), heading_yaw(prev[:, 3:7])
    c, s = torch.cos(yaw0), torch.sin(yaw0)
    d = (cur[:, :2] - prev[:, :2]) / (REPLAN * 0.02)
    v0 = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    w0 = torch.atan2(torch.sin(yaw0 - yawp), torch.cos(yaw0 - yawp)) / (REPLAN * 0.02)
    lam = math.log(2) / VEL_HALFLIFE
    p = torch.zeros_like(v0)
    th = torch.zeros_like(w0)
    feats = []
    for k in range(1, max(CMD_OFFSETS) + 1):
        a = math.exp(-lam * k * 0.02)
        v = cmd[:, :2] + (v0 - cmd[:, :2]) * a
        w = cmd[:, 2] + (w0 - cmd[:, 2]) * a
        th = th + w * 0.02
        p = p + 0.02 * torch.stack([torch.cos(th) * v[:, 0] - torch.sin(th) * v[:, 1], torch.sin(th) * v[:, 0] + torch.cos(th) * v[:, 1]], -1)
        if k in CMD_OFFSETS:
            feats.append(torch.cat([p, torch.cos(th)[:, None], torch.sin(th)[:, None]], -1))
    return torch.cat(feats, -1)


@torch.no_grad()
def rollout(model, diff, scan, tiles, start, cmd, seconds, steps, generator=None, guidance=1.0):
    """tiles (B,), start (B, 25), cmd (B, 3) -> (B, T, 25) frames at 50 Hz (terrain frame)."""
    H, B = model.cfg.hist, start.shape[0]
    n = model.norm
    traj = start[:, None].repeat(1, max(H, REPLAN + 1), 1)
    while traj.shape[1] < seconds * 50 + H:
        cur = traj[:, -1]
        origin, yaw0 = cur[:, :3], heading_yaw(cur[:, 3:7])
        h = traj[:, -H:]
        hist = to_canonical(h[..., :3], h[..., 3:7], h[..., 7:], origin, yaw0)
        st = torch.zeros(B, 49, device=cur.device)
        st[:, :7] = cur[:, :7]
        hmap = (scan(st, tiles) - origin[:, 2:3]).reshape(B, *GRID_SHAPE)
        c = command_path(traj, operator(traj, cmd))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = diff.sample(model, n.frames(hist), n.hmap(hmap), n.cmd(c), torch.ones(B, device=cur.device), steps=steps, guidance=guidance, generator=generator).float()
        pos, quat, q = from_canonical(n.frames_inv(x[:, :REPLAN]), origin, yaw0)
        traj = torch.cat([traj, torch.cat([pos, quat, q], -1)], 1)
    return traj[:, H - 1 :]


def score(sc, terr, traj, tiles, feet, S):
    """Per scene (S rollouts each, scene-major in traj (B, T, 25)): success and its ingredients."""
    res = {}
    total = 0
    for k, s in enumerate(sc):
        tr = traj[k * S : (k + 1) * S]
        tl = tiles[k * S : (k + 1) * S]
        fm = feet(tr[..., :3], tr[..., 3:7], tr[..., 7:], tl)
        sole, gap = feet.gap(tr[..., :3], tr[..., 3:7], tr[..., 7:], tl)
        pen = (-gap).clamp_min(0).amax((1, 2)).cpu().numpy()  # (S,)
        mid = sole.mean(-2)
        hs = ((mid[:, 1:, :, :2] - mid[:, :-1, :, :2]).norm(dim=-1) / 0.02)
        con = (gap[:, 1:] < 0.01).float()
        slide = ((hs * con).sum((1, 2)) / con.sum((1, 2)).clamp_min(1)).cpu().numpy()  # (S,) m/s, feet in contact
        ground = np.array([[terr[k].h(p[:2]) for p in row] for row in tr[:, :, :3].cpu().numpy()])  # (S, T)
        above = tr[:, :, 2].cpu().numpy() - ground
        end_x = tr[:, -1, 0].cpu().numpy()
        # mean planar velocity and yaw rate over the second half
        half = tr.shape[1] // 2
        span = (tr.shape[1] - 1 - half) * 0.02
        v = ((tr[:, -1, :2] - tr[:, half, :2]) / span).cpu().numpy()
        dyaw = heading_yaw(tr[:, -1, 3:7]) - heading_yaw(tr[:, half, 3:7])
        yaw_rate = (torch.atan2(torch.sin(dyaw), torch.cos(dyaw)) / span).cpu().numpy()
        speed = np.linalg.norm(v, axis=1)
        yaw_all = heading_yaw(tr[..., 3:7])
        yaw_dev = np.degrees(torch.atan2(torch.sin(yaw_all), torch.cos(yaw_all)).abs().amax(1).cpu().numpy())  # from the start heading
        # sliding is reported, not scored: mocap flat walking itself reads ~0.5 m/s here (low swing clearance)
        # floating: root no more than 1.1 m above the ground, plus the drop of a step-down (the body crosses a ledge
        # before it descends)
        drop = max(0.0, float(terr[k].h(np.zeros(2))) - s.top)
        ok = (pen < 0.04) & (above.min(1) > s.min_above) & (above.max(1) < 1.1 + drop) & (np.abs(ground[:, -1] - s.top) < 0.02)
        ok &= speed < 1.5 * np.hypot(*s.cmd[:2]) + 0.2
        if s.past is not None:
            ok &= end_x > s.past
        else:  # speed within 30 % of the command, turning within 50 %
            ok &= np.abs(speed - np.hypot(*s.cmd[:2])) < 0.3 * np.hypot(*s.cmd[:2])
            if s.cmd[2] != 0:
                ok &= np.abs(yaw_rate - s.cmd[2]) < 0.5 * abs(s.cmd[2])
        total += int(ok.sum())
        res[s.name] = {"success": f"{int(ok.sum())}/{S}", **{m: round(v_, 3) for m, v_ in fm.items()}, "max_pen_cm": (pen * 100).round(1).tolist(),
                       "end_x_m": end_x.round(2).tolist(), "end_ground_m": ground[:, -1].round(2).tolist(),
                       "min_root_above_ground_m": above.min(1).round(2).tolist(), "speed_2nd_half_m_s": speed.round(2).tolist(),
                       "slide_m_s": slide.round(2).tolist(), "yaw_rate_2nd_half": yaw_rate.round(2).tolist(),
                       "max_heading_dev_deg": yaw_dev.round(0).tolist()}
        print(f"{s.name}: {res[s.name]}", flush=True)
    print(f"success {total}/{len(sc) * S}")
    return res, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance on the command")
    ap.add_argument("--steps", type=int, default=20, help="DDIM steps per plan")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, diff, ck = load(a.ckpt, a.device)
    sc = scenes()
    terr = [Terrain(s.hf, np.array([X0, Y0]), DX) for s in sc]
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(a.device)
    S = a.samples
    tiles = torch.arange(len(sc), device=a.device).repeat_interleave(S)
    st = standing_frame(ck["args"]["library"], a.device)
    start = st.repeat(len(tiles), 1)
    start[:, 2] += torch.as_tensor(np.array([t.h(np.zeros(2)) for t in terr]), dtype=torch.float32, device=a.device).repeat_interleave(S)
    cmd = torch.as_tensor(np.array([s.cmd for s in sc]), dtype=torch.float32, device=a.device).repeat_interleave(S, 0)
    g = torch.Generator(device=a.device).manual_seed(0)
    traj = rollout(model, diff, scan, tiles, start, cmd, a.seconds, a.steps, generator=g, guidance=a.guidance)
    feet = FootMetrics(scan, a.device)
    res, total = score(sc, terr, traj, tiles, feet, S)
    (out / "results.json").write_text(json.dumps({"ckpt": a.ckpt, "step": ck["step"], "guidance": a.guidance, "success": f"{total}/{len(sc) * S}", "scenes": res}, indent=1))
    torch.save({"traj": traj.cpu(), "scenes": [s.name for s in sc], "samples": S}, out / "traj.pt")
    if a.render:
        render_all(sc, terr, traj[::S].cpu().numpy(), out)


def render_all(sc, terr, Q, out: Path, w=640, h=360):
    sys.path.insert(0, str(REPO / "scripts/parc"))
    from render_hq import FPS, camera_track, dash_model, encode, render, resample

    vids = []
    for k, s in enumerate(sc):
        boxes = np.array([[0, 0, terr[k].floor - 0.05, 0.01, 0.01, 0.05]] + [list(b) for b in terr[k].boxes()])
        m = dash_model(boxes, w, h)
        for g in range(m.ngeom):  # terrain in a plain colour: with the floor's checker texture, tops vanish into it
            if m.geom(g).name.startswith("terrain_"):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (0.60, 0.66, 0.78, 1.0)
        q = resample(Q[k].astype(np.float64), 50, 1.0)
        look, az = camera_track(q, FPS, -0.15)
        p = out / f"{k:02d}.mp4"
        encode(str(p), render(m, q, look, az, 3.6, w, h, label=s.name), w, h)
        vids.append(p)
    cols = 4
    rows = math.ceil(len(vids) / cols)
    inputs = sum([["-i", str(v)] for v in vids], [])
    pad = cols * rows - len(vids)
    filt = "".join(f"[{i}:v]" for i in range(len(vids)))
    if pad:
        inputs += ["-f", "lavfi", "-i", f"color=c=white:s={w}x{h}:r={FPS}"] * pad
        filt = "".join(f"[{i}:v]" for i in range(len(vids) + pad))
    layout = "|".join(f"{c * w}_{r * h}" for r in range(rows) for c in range(cols))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", f"{filt}xstack=inputs={len(vids) + pad}:layout={layout}:shortest=1",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out / "grid.mp4")], check=True)


if __name__ == "__main__":
    main()
