"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.policy_trainer import PolicyTrainer, VelocityImagination
from supermimic.interfaces.velocity import CommandSampler, fall_features, reference_commands, velocity_of, velocity_reward_terms
from supermimic.kinematics.fk import Kinematics
from supermimic.policy.nets import Critic, Policy, VelocityCritic
from supermimic.robots import get_robot
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.robot = spec = get_robot(cfg.robot)
        cfg.policy.num_joints = spec.num_joints
        cfg.policy.goal_dim = cfg.env.future_frames * spec.goal_feat_dim + (9 if cfg.env.anchor_obs else 0)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        # With hardware DR, its parameters (per joint and per body) enter xi (envs/sim_context.py).
        cfg.world_model.state_spec.xi_hw_bodies = spec.num_bodies if cfg.env.hardware_dr else 0
        vcfg = cfg.velocity
        cfg.policy.velocity = vcfg.enabled
        cfg.policy.cmd_scale = vcfg.cmd_scale
        if vcfg.enabled:
            assert not cfg.env.anchor_obs, "velocity interface needs a proprioception-only policy"
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot=spec.name,
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                hardware_dr=cfg.env.hardware_dr,
                command_delay=cfg.env.command_delay,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                fall_height=vcfg.fall_height,
                fall_tilt=vcfg.fall_tilt,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = Kinematics(spec.load_model(), device=self.device)
        heads = AnalyticHeads(fk, spec.tracked_bodies, spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy, xi_dim=self.wm.spec.xi_dim).to(self.device)
        assert self.buffer.xi.shape[-1] == self.wm.spec.xi_dim, (self.buffer.xi.shape, self.wm.spec.xi_dim)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)

        # Velocity interface: the last env_fraction of the envs follow velocity commands.
        self.vel = None
        if vcfg.enabled:
            n = env.num_envs
            vel_mask = torch.arange(n, device=self.device) >= n - int(round(n * vcfg.env_fraction))
            self.command.track_mask[vel_mask] = False
            self.vel_mask, self.vel_ids = vel_mask, vel_mask.nonzero().squeeze(-1)
            self.ref_cmd = reference_commands(self.lib, vcfg.ref_window)
            # Velocity envs start in upright library frames (root high, small tilt).
            up = []
            for i in range(0, self.lib.num_frames, 1 << 22):
                f = torch.arange(i, min(i + (1 << 22), self.lib.num_frames), device=self.device)
                st = self.lib.ref_state(f)
                feats = fall_features(st)
                up.append(f[(feats[:, 0] > vcfg.fall_height + 0.25) & (feats[:, 1] < 0.3)])
            self.command.untracked_frames = torch.cat(up)
            self.vel_sampler = CommandSampler(len(self.vel_ids), self.ref_cmd, vcfg, self.device)
            self.vel_critic = VelocityCritic(cfg.critic, cfg.policy, self.wm.spec.xi_dim).to(self.device)
            broadcast_params(self.vel_critic)
            self.vel = VelocityImagination(vcfg, self.vel_critic, self.ref_cmd, self.command.track_mask, CommandSampler(1, self.ref_cmd, vcfg, self.device))
            self.wm_trainer.track_mask = self.command.track_mask
            self.vel_ep_sum = torch.zeros(len(self.vel_ids), device=self.device)
            self.vel_ep_len = torch.zeros(len(self.vel_ids), device=self.device)
            self.prev_action = torch.zeros(n, spec.num_joints, device=self.device)
            if self.is_main:
                print(f"[INFO] velocity interface: {len(self.vel_ids)} envs, {self.command.untracked_frames.numel()} upright reset frames, cmd ranges vx {vcfg.vx_range} vy {vcfg.vy_range} wz {vcfg.wz_range}")
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer, velocity=self.vel)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.start_iteration = 0  # warm-up (buffer refill) counts from here, also after a resume
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)
        elif cfg.init_from is not None:
            self.init_from(cfg.init_from)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
                **({"vel_critic": self.vel_critic.state_dict(), "vel_critic_opt": self.policy_trainer.vel_critic_opt.state_dict()} if self.vel is not None else {}),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        ck_robot = ck["config"].get("robot", "g1")
        assert ck_robot == self.robot.name, f"checkpoint is for {ck_robot}, run is for {self.robot.name}"
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        if self.vel is not None:
            self.vel_critic.load_state_dict(ck["vel_critic"])
            self.policy_trainer.vel_critic_opt.load_state_dict(ck["vel_critic_opt"])
        self.iteration = ck["iteration"]
        self.start_iteration = self.iteration
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    def init_from(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        ck_robot = ck["config"].get("robot", "g1")
        assert ck_robot == self.robot.name, f"checkpoint is for {ck_robot}, run is for {self.robot.name}"
        self.wm.load_state_dict(ck["wm"])
        missing, unexpected = self.policy.load_state_dict(ck["policy"], strict=False)
        assert not unexpected and all(k.startswith("encoders.velocity.") for k in missing), (missing, unexpected)
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        if self.is_main:
            print(f"[INFO] initialized from {path} (iteration {ck['iteration']}); new: {sorted({'.'.join(k.split('.')[:2]) for k in missing})}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        if self.vel is None:
            return self.policy(obs["policy"], obs["goal"])
        o = obs["policy"]
        a = self.policy(o, obs["goal"])
        a[self.vel_ids] = self.policy(o[self.vel_ids], self.vel_sampler.cmd, "velocity")
        return a

    @torch.no_grad()
    def _velocity_after_step(self, action: torch.Tensor, done: torch.Tensor) -> list[float]:
        """Velocity reward of the step just taken (envs that did not reset), episode bookkeeping and
        command updates. Returns normalized returns of finished velocity episodes."""
        ids, vcfg = self.vel_ids, self.cfg.velocity
        s = self.recorder.state_io.read()[ids]
        d = done[ids]
        r = velocity_reward_terms(s, action[ids], self.prev_action[ids], self.vel_sampler.cmd, self.wm.heads.soft_limits, vcfg)
        live = ~d
        self.vel_ep_sum += torch.where(live, r["total"], torch.zeros_like(r["total"]))
        self.vel_ep_len += live.float()
        err = velocity_of(s) - self.vel_sampler.cmd
        self._vel_stats["lin_err"] += float((err[:, 0:2].norm(dim=-1) * live).sum())
        self._vel_stats["yaw_err"] += float((err[:, 2].abs() * live).sum())
        self._vel_stats["n"] += int(live.sum())
        out = []
        if d.any():
            j = d.nonzero().squeeze(-1)
            out = (self.vel_ep_sum[j] / (vcfg.max_step_reward * self.vel_ep_len[j].clamp_min(1))).tolist()
            self._vel_stats["falls"] += int(self.env.termination_manager.terminated[ids[j]].sum())
            self._vel_stats["episodes"] += int(j.numel())
            self.vel_ep_sum[j] = 0.0
            self.vel_ep_len[j] = 0.0
            self.vel_sampler.resample(j, self.command.frame[ids[j]])
        self.prev_action.copy_(action)
        self.prev_action[done] = 0.0
        self.vel_sampler.step(self.command.frame[ids], clip_over=self.command.clip_end[ids])
        return out

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        if self.vel is not None:
            self.vel_sampler.resample(torch.arange(len(self.vel_ids), device=self.device), self.command.frame[self.vel_ids])
            self._vel_stats = {"lin_err": 0.0, "yaw_err": 0.0, "n": 0, "falls": 0, "episodes": 0}
            vel_returns: list[float] = []
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if self.vel is not None:
                vel_returns.extend(self._velocity_after_step(action, done))
            track_done = done if self.vel is None else done & ~self.vel_mask
            if track_done.any():
                ids = track_done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    if self.vel is not None:
                        vs = self._vel_stats
                        if vel_returns:
                            self.writer.add_scalar("train/vel_episode_norm_return", sum(vel_returns) / len(vel_returns), it)
                        if vs["n"]:
                            self.writer.add_scalar("train/vel_lin_err", vs["lin_err"] / vs["n"], it)
                            self.writer.add_scalar("train/vel_yaw_err", vs["yaw_err"] / vs["n"], it)
                        if vs["episodes"]:
                            self.writer.add_scalar("train/vel_fall_rate", vs["falls"] / vs["episodes"], it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    if self.vel is not None:
                        vs = self._vel_stats
                        msg += (
                            f" | vel ret {sum(vel_returns) / max(len(vel_returns), 1):.3f} (n={len(vel_returns)}) "
                            f"err {vs['lin_err'] / max(vs['n'], 1):.2f} m/s {vs['yaw_err'] / max(vs['n'], 1):.2f} rad/s falls {vs['falls']}/{vs['episodes']} "
                            f"imag {agg.get('policy/vel_imag_return_undisc', float('nan'))/max(agg_n,1):.3f} "
                            f"surv {agg.get('policy/vel_imag_survival_H', float('nan'))/max(agg_n,1):.3f} "
                            f"align {agg.get('policy/vel_align_loss', float('nan'))/max(agg_n,1):.3f}"
                        )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
                if self.vel is not None:
                    vel_returns = []
                    self._vel_stats = {"lin_err": 0.0, "yaw_err": 0.0, "n": 0, "falls": 0, "episodes": 0}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
