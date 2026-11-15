"""Forward depth from Dash's torso RealSense D435: the depth student's camera.

Ported from mjlab_dash (``tasks/velocity/config/dash/depth.py``, commit db83fd6), which follows the
Go1 setup of Zhuang et al.'s parkour code (github.com/ZiwenZhuang/parkour, go1_field_distill_config.py,
a D435i: the same depth hardware). The camera itself (``realsense_depth``, pose and FOV) is added to
the robot by ``assets/dash/dash.py`` (``realsense=True``).

``process_depth`` is the one preprocessing path for sim and hardware: on the robot, 424x240 depth goes
through RealSense's spatial + temporal filters (``ONBOARD_FILTERS``), is converted to metres and passed
to it, so the network sees identically cropped, clipped and resized images in both.

Added here: stereo dropouts (``_dropouts``). A stereo camera loses depth along depth edges (one imager
sees past the edge, the other does not) and in blobs (no texture, glare); the onboard hole filling
closes only small holes. Stair edges are what the student must see, so it learns with them missing.
The rates are guesses until compared with recorded robot depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg

CAMERA = "robot/realsense_depth"
SENSOR = "forward_depth"


@dataclass(frozen=True)
class DepthCfg:
    # Hardware stream: 424x240, the parkour Go1 setting. The lower resolution lets the stereo match
    # closer (0.12 m vs 0.175 m at 640x480), which matters with the ground this near the lens.
    stream_width: int = 424
    stream_height: int = 240
    # Sim renders at a quarter of that, with the same aspect and FOV.
    render_width: int = 106
    render_height: int = 60
    # Columns dropped from each side, as fractions of the width. The left band is the D435's
    # left-imager-only region with no stereo match (parkour Go1: 60 and 46 px of 424).
    crop_left: float = 60 / 424
    crop_right: float = 46 / 424
    out_height: int = 48
    out_width: int = 64
    depth_range: tuple[float, float] = (0.0, 2.0)
    # Stereo noise, parkour Go1 values (m): uniform in [0, noise) added to every pixel, and no depth
    # (0) closer than min_distance.
    min_distance: float = 0.12
    far_distance: float = 2.0
    near_noise: float = 0.02
    far_noise: float = 0.08
    # Dropouts (render pixels, set to 0 = no depth). Edge: a pixel whose inverse depth has a second
    # difference above edge_jump (1/m) along a row or a column is lost with probability edge_drop. Inverse
    # planar depth is affine in the image on any plane, so floors, treads and risers never count however
    # grazing the view; occluding edges do (a 10 cm step down 2 m ahead: ~0.07; plane creases: <~0.04).
    # Blobs: with probability blob_prob a frame loses 1..blob_max rectangles of blob_size px a side (at
    # 106x60; x4 at 424x240, past the onboard hole filling's 16 px).
    edge_jump: float = 0.06
    edge_drop: float = 0.3
    blob_prob: float = 0.3
    blob_max: int = 3
    blob_size: tuple[int, int] = (3, 10)
    # Frames arrive at 10 Hz with 80-140 ms latency at the 50 Hz policy rate. Latency is a placeholder
    # (parkour Go2, USB 3): measure it on the robot.
    refresh_steps: int = 5
    latency_steps: tuple[int, int] = (4, 7)


DEPTH = DepthCfg()

# The robot side's RealSense post-processing, for the deploy code (not used in sim).
ONBOARD_FILTERS = {
    "spatial": {"filter_magnitude": 5, "filter_smooth_alpha": 0.75, "filter_smooth_delta": 1},
    "spatial_holes_fill": 4,
    "temporal": {"filter_smooth_alpha": 0.75, "filter_smooth_delta": 1},
}


def process_depth(depth: torch.Tensor, cfg: DepthCfg = DEPTH) -> torch.Tensor:
    """Metric depth (B, H, W) -> network input (B, 1, out_h, out_w) in [0, 1]. Invalid pixels (0) stay
    at 0, the D435's no-depth value."""
    lo, hi = cfg.depth_range
    x = ((depth.clamp(lo, hi) - lo) / (hi - lo)).unsqueeze(1)
    w = x.shape[-1]
    x = x[..., round(cfg.crop_left * w) : w - round(cfg.crop_right * w)]
    x = F.interpolate(x, size=(cfg.out_height, cfg.out_width), mode="bilinear")
    return x.clamp(0.0, 1.0)


def _dropouts(d: torch.Tensor, cfg: DepthCfg) -> torch.Tensor:
    """(B, H, W) clean metric depth -> (B, H, W) bool, the pixels the stereo loses: at occluding edges and
    in blobs."""
    b, h, w = d.shape
    inv = 1.0 / d.clamp_min(0.05)
    edge = torch.zeros_like(d, dtype=torch.bool)
    edge[:, :, 1:-1] |= (inv[:, :, 2:] - 2 * inv[:, :, 1:-1] + inv[:, :, :-2]).abs() > cfg.edge_jump
    edge[:, 1:-1] |= (inv[:, 2:] - 2 * inv[:, 1:-1] + inv[:, :-2]).abs() > cfg.edge_jump
    lost = edge & (torch.rand_like(d) < cfg.edge_drop)
    lo, hi = cfg.blob_size
    ys = torch.arange(h, device=d.device)[None, :, None]
    xs = torch.arange(w, device=d.device)[None, None, :]
    count = torch.where(torch.rand(b, device=d.device) < cfg.blob_prob, torch.randint(1, cfg.blob_max + 1, (b,), device=d.device), 0)
    for k in range(cfg.blob_max):
        on = count > k
        sh = torch.randint(lo, hi + 1, (b, 2), device=d.device)
        y0 = (torch.rand(b, device=d.device) * h).long()
        x0 = (torch.rand(b, device=d.device) * w).long()
        box = ((ys >= y0[:, None, None]) & (ys < (y0 + sh[:, 0])[:, None, None]) & (xs >= x0[:, None, None]) & (xs < (x0 + sh[:, 1])[:, None, None]))
        lost |= box & on[:, None, None]
    return lost


def corrupt_depth(d: torch.Tensor, cfg: DepthCfg = DEPTH) -> torch.Tensor:
    """(B, H, W) clean metric depth -> what the D435 delivers: noise, and 0 where too close or lost."""
    lost = _dropouts(d, cfg) | (d < cfg.min_distance)
    noise = torch.rand_like(d) * torch.where(d > cfg.far_distance, cfg.far_noise, cfg.near_noise)
    return torch.where(lost, torch.zeros_like(d), d + noise)


class forward_depth:
    """Noisy 10 Hz forward depth from the RealSense camera sensor, (N, out_h * out_w) in [0, 1].

    Between refreshes the last frame is held, as on hardware; refresh phases are staggered across envs.
    Latency is added by the observation manager's delay buffer (``depth_observation_group``).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self._cfg: DepthCfg = cfg.params.get("depth_cfg", DEPTH)
        n = env.num_envs
        self._phase = torch.randint(0, self._cfg.refresh_steps, (n,), device=env.device)
        self._step = 0
        self._frame = torch.zeros(n, 1, self._cfg.out_height, self._cfg.out_width, device=env.device)
        self._fresh = torch.ones(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRlEnv, depth_cfg: DepthCfg = DEPTH, corrupt: bool = True) -> torch.Tensor:
        del depth_cfg
        sensor: CameraSensor = env.scene[SENSOR]
        assert sensor.data.depth is not None
        due = ((self._step + self._phase) % self._cfg.refresh_steps == 0) | self._fresh
        self._step += 1
        if due.any():
            d = sensor.data.depth[due, ..., 0]
            # The renderer returns 0 where a ray hits nothing: open space, so far, not the D435's
            # 0 = no stereo match (too close or a dropout).
            d = torch.where(d > 0, d, self._cfg.depth_range[1])
            if corrupt:
                d = corrupt_depth(d, self._cfg)
            self._frame[due] = process_depth(d, self._cfg)
            self._fresh[due] = False
        return self._frame.flatten(1)


def depth_camera_cfg(cfg: DepthCfg = DEPTH) -> CameraSensorCfg:
    """The D435 depth sensor. Renders terrain (group 0) and the robot's visual meshes (2); not group 1,
    the camera's own housing, nor the oversized group-3 colliders, which the real lens can't see."""
    return CameraSensorCfg(name=SENSOR, camera_name=CAMERA, width=cfg.render_width, height=cfg.render_height, data_types=("depth",),
                           enabled_geom_groups=(0, 2), use_textures=False, use_shadows=False)


def depth_observation_group(noisy: bool, cfg: DepthCfg = DEPTH) -> ObservationGroupCfg:
    """The ``depth`` group; noisy: stereo noise, dropouts and the 80-140 ms latency (else clean, no lag)."""
    lo, hi = cfg.latency_steps
    term = ObservationTermCfg(func=forward_depth, params={"depth_cfg": cfg, "corrupt": noisy}, delay_min_lag=lo if noisy else 0,
                              delay_max_lag=hi if noisy else 0, delay_update_period=cfg.refresh_steps)
    return ObservationGroupCfg(terms={"forward_depth": term}, concatenate_terms=True, enable_corruption=False)


def add_camera_randomization(events: dict[str, EventTermCfg]) -> None:
    """Mount tolerance of a printed jig (a few mm, ~1 deg) and per-unit FOV spread (this unit: 58.9 deg)."""
    cam = SceneEntityCfg("robot", camera_names=("realsense_depth",))
    events["realsense_pos"] = EventTermCfg(func=dr.cam_pos, mode="startup",
                                           params={"asset_cfg": cam, "operation": "add", "ranges": {0: (-0.003, 0.003), 1: (-0.003, 0.003), 2: (-0.003, 0.003)}})
    events["realsense_quat"] = EventTermCfg(func=dr.cam_quat, mode="startup",
                                            params={"asset_cfg": cam, "roll_range": (-0.01, 0.01), "pitch_range": (-0.02, 0.02), "yaw_range": (-0.01, 0.01)})
    events["realsense_fovy"] = EventTermCfg(func=dr.cam_fovy, mode="startup", params={"asset_cfg": cam, "operation": "add", "ranges": (-1.0, 1.0)})
