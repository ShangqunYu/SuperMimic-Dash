"""Robot registry: everything robot-specific the pipeline needs, in one place.

Every network / state dimension follows from the number of actuated hinge joints NJ:

  state  s  = [root_pos 3, root_quat 4, joint_pos NJ, root_lin_vel_w 3, root_ang_vel_b 3, joint_vel NJ]
            = qpos (7 + NJ) | qvel (6 + NJ)                                         13 + 2 NJ
  xi        = [foot friction 1, encoder bias NJ, COM offset 3]                       4 + NJ
  obs frame = [joint_pos NJ, joint_vel NJ, base_ang_vel 3, projected_gravity 3, last_action NJ]
                                                                                     3 NJ + 6
  goal frame = [joint_pos - q0 NJ, joint_vel NJ, grav 3, ang_vel_b 3, lin_vel_h 3, height 1]
                                                                                     2 NJ + 10
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

from supermimic.assets.dash import dash as dash_asset


def state_dim(nj: int) -> int:
    return 13 + 2 * nj


def xi_dim(nj: int) -> int:
    return 4 + nj


def obs_dim(nj: int) -> int:
    return 3 * nj + 6


def goal_feat_dim(nj: int) -> int:
    return 2 * nj + 10


def nj_from_state_dim(d: int) -> int:
    assert d >= 13 and (d - 13) % 2 == 0, f"not a state dimension: {d}"
    return (d - 13) // 2


def nj_from_obs_dim(d: int) -> int:
    assert d >= 6 and (d - 6) % 3 == 0, f"not an observation dimension: {d}"
    return (d - 6) // 3


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]  # MuJoCo joint order (= qpos order = action order)
    xml_path: Callable[[], Path]
    entity_cfg: Callable[..., EntityCfg]  # kwargs: command_delay (robots that model one)
    action_scale: Callable[[], dict[str, float]]
    root_body: str  # free-joint body (the state's root pose)
    anchor_body: str  # reward / re-anchoring body, one of tracked_bodies
    tracked_bodies: tuple[str, ...]
    foot_geom_regex: str  # geoms whose friction is randomized and enters xi
    com_body: str  # body whose COM offset is randomized and enters xi
    nconmax: int | None
    njmax: int | None
    # Adds hardware-derived randomization to an events dict (None: robot has none).
    hardware_dr: Callable[[dict], None] | None = None

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return state_dim(self.num_joints)

    @property
    def xi_dim(self) -> int:
        return xi_dim(self.num_joints)

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.num_joints)

    @property
    def goal_feat_dim(self) -> int:
        return goal_feat_dim(self.num_joints)

    def load_model(self) -> mujoco.MjModel:
        """Standalone CPU MjModel. Asserts a free root followed by the expected hinge order."""
        model = mujoco.MjModel.from_xml_path(str(self.xml_path()))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
        assert tuple(names[1:]) == self.joint_names, f"{self.name}: joint order mismatch: {names[1:]}"
        assert model.nq == 7 + self.num_joints and model.nv == 6 + self.num_joints
        root = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
        assert root == self.root_body, f"{self.name}: first body is {root}, spec says {self.root_body}"
        return model

    def body_names(self) -> tuple[str, ...]:
        """All bodies in model order (world excluded): the body axis of our npz files."""
        model = mujoco.MjModel.from_xml_path(str(self.xml_path()))
        return tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody))


# ----------------------------------------------------------------------------- G1

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _g1_xml_path() -> Path:
    import mjlab

    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _g1_entity_cfg() -> EntityCfg:
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _g1_action_scale() -> dict[str, float]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    return G1_ACTION_SCALE


G1 = RobotSpec(
    name="g1",
    joint_names=G1_JOINT_NAMES,
    xml_path=_g1_xml_path,
    entity_cfg=_g1_entity_cfg,
    action_scale=_g1_action_scale,
    root_body="pelvis",
    anchor_body="torso_link",
    tracked_bodies=(
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
    nconmax=35,
    njmax=250,
)


# ----------------------------------------------------------------------------- Dash


DASH = RobotSpec(
    name="dash",
    joint_names=dash_asset.DASH_JOINT_NAMES,
    xml_path=lambda: dash_asset.DASH_XML,
    entity_cfg=dash_asset.get_dash_robot_cfg,
    action_scale=lambda: dash_asset.DASH_ACTION_SCALE,
    # The torso is both the free-joint root and the IMU body, so it is also the anchor.
    root_body="torso",
    anchor_body="torso",
    # One body per G1 counterpart (as in mjlab_dash's tracking task): dist_hip ~
    # hip_roll_link, lower_leg ~ knee_link, foot ~ ankle_roll_link, upper_arm ~
    # shoulder_roll_link, lower_arm ~ elbow_link. No wrist: lower_arm is the arm's end.
    tracked_bodies=(
        "torso",
        "l_dist_hip",
        "l_lower_leg",
        "l_foot",
        "r_dist_hip",
        "r_lower_leg",
        "r_foot",
        "l_upper_arm",
        "l_lower_arm",
        "r_upper_arm",
        "r_lower_arm",
    ),
    foot_geom_regex=dash_asset.FOOT_GEOM_REGEX,
    com_body="torso",
    # Full self-collision (capsules + forearm-thigh hulls): mjlab_dash's contact budget.
    nconmax=None,
    njmax=300,
    hardware_dr=dash_asset.add_hardware_randomization,
)


ROBOTS: dict[str, RobotSpec] = {"g1": G1, "dash": DASH}


def get_robot(name: str) -> RobotSpec:
    try:
        return ROBOTS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}") from None
