"""Depth student: the one network that runs on Dash (i7 UP Xtreme, CPU), distilled by DAgger from the generator +
v6 teacher (scripts/student/dagger.py).

Inputs, all available on the robot: proprioception (the ``proprio`` observation group: joint positions and
velocities, angular velocity, gravity, last action) and the joystick command (vx, vy, wz) each 50 Hz step, and the
forward depth image (``depth`` group, 64x48 in [0, 1]). A small transformer attends over a fixed window: one token
per depth frame (``n_depth``, taken every ``depth_every`` steps: 2 s) and one per step of proprioception + command
(``n_proprio``: 0.5 s), plus a readout token; out comes the joint-position action (the teacher's action space).
A fixed window, not a recurrent state: DAgger trains on shuffled self-contained windows, and the deployed policy
is a plain function of its recent inputs. Each depth frame is encoded once when it arrives (``encode_depth``) and
its token reused until it leaves the window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class StudentCfg:
    proprio_dim: int = 60
    cmd_dim: int = 3
    act_dim: int = 18
    depth_h: int = 48
    depth_w: int = 64
    n_depth: int = 20
    depth_every: int = 5  # control steps between depth tokens (the camera's 10 Hz)
    n_proprio: int = 25
    width: int = 128
    layers: int = 3
    heads: int = 4
    ff: int = 256
    aux_dim: int = 0  # > 0: a training-only head predicting the terrain scan around the robot (rl_finetune.py --aux)


class Student(nn.Module):
    def __init__(self, cfg: StudentCfg):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * ((cfg.depth_h + 7) // 8) * ((cfg.depth_w + 7) // 8), w),
        )  # fmt: skip
        d_in = cfg.proprio_dim + cfg.cmd_dim
        self.proprio = nn.Sequential(nn.Linear(d_in, w), nn.ELU(), nn.Linear(w, w))
        self.readout = nn.Parameter(torch.zeros(1, 1, w))
        self.pos = nn.Parameter(torch.randn(1, 1 + cfg.n_depth + cfg.n_proprio, w) * 0.02)  # slot embeddings (type + age)
        layer = nn.TransformerEncoderLayer(w, cfg.heads, cfg.ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 256), nn.ELU(), nn.Linear(256, cfg.act_dim))
        if cfg.aux_dim:
            self.aux = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 256), nn.ELU(), nn.Linear(256, cfg.aux_dim))
        # Input normalization of proprioception + command, set once from the first data (set_normalization).
        self.register_buffer("in_mean", torch.zeros(d_in))
        self.register_buffer("in_std", torch.ones(d_in))

    def set_normalization(self, x: torch.Tensor) -> None:
        """x (..., proprio_dim + cmd_dim) samples of the inputs."""
        x = x.reshape(-1, x.shape[-1]).float()
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0).clamp_min(1e-2))

    def encode_depth(self, img: torch.Tensor) -> torch.Tensor:
        """img (B, depth_h * depth_w) or (B, 1, depth_h, depth_w) in [0, 1] -> (B, width) token."""
        return self.cnn(img.reshape(-1, 1, self.cfg.depth_h, self.cfg.depth_w))

    def forward(self, proprio: torch.Tensor, depth_tok: torch.Tensor, aux: bool = False):
        """proprio (B, n_proprio, proprio_dim + cmd_dim) oldest first; depth_tok (B, n_depth, width) oldest first
        -> action (B, act_dim); aux: (action, terrain scan prediction (B, aux_dim))."""
        h = self.trunk(proprio, depth_tok)
        return (self.head(h), self.aux(h)) if aux else self.head(h)

    def trunk(self, proprio: torch.Tensor, depth_tok: torch.Tensor) -> torch.Tensor:
        """-> the readout token's features (B, width), the heads' input."""
        p = self.proprio((proprio - self.in_mean) / self.in_std)
        x = torch.cat([self.readout.expand(len(p), -1, -1), depth_tok, p], 1) + self.pos
        return self.encoder(x)[:, 0]

    def config(self) -> dict:
        return asdict(self.cfg)


class StudentRunner:
    """Runs the student step by step as on the robot: keeps the proprioception and depth-token windows (filled
    with the first input at the start) and takes a depth token every ``depth_every`` steps."""

    def __init__(self, model: Student, n: int, device: str):
        self.m = model
        self.n = n
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.P: torch.Tensor | None = None
        self.D: torch.Tensor | None = None

    @torch.no_grad()
    def act(self, proprio: torch.Tensor, cmd: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """proprio (n, proprio_dim), cmd (n, 3), depth (n, depth_h * depth_w) -> action (n, act_dim)."""
        c = self.m.cfg
        p = torch.cat([proprio, cmd], -1)
        self.P = p[:, None].repeat(1, c.n_proprio, 1) if self.P is None else torch.cat([self.P[:, 1:], p[:, None]], 1)
        if self.t % c.depth_every == 0:
            tok = self.m.encode_depth(depth)
            self.D = tok[:, None].repeat(1, c.n_depth, 1) if self.D is None else torch.cat([self.D[:, 1:], tok[:, None]], 1)
        self.t += 1
        return self.m(self.P, self.D)


def load_student(path: str, device: str) -> Student:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Student(StudentCfg(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()
