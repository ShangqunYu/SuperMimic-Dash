"""Dash humanoid, vendored from mjlab_dash (``src/mjlab/asset_zoo/robots/dash``, commit 0d56f20).

18 DOF: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch) and 4 per arm (shoulder
pitch/roll/yaw, elbow pitch); the torso is the free-joint root and carries the IMU.

mjlab_dash pins an older mujoco-warp, so the XML, meshes and constants are copied here
rather than depended on. Differences from the source:
  * the forearm-thigh mesh hulls of mjlab_dash's tracking task are always added
    (``get_spec``): mocap drives the hands into the thighs;
  * the actuator command delay is off unless asked for (``get_dash_robot_cfg``): a delay
    longer than one 20 ms control step makes s' depend on the previous action, which the
    world model does not see. It belongs with the hardware randomization.
  * the RealSense D435 + jig (mjlab_dash commit db83fd6) is added only when asked for
    (``realsense``): the teachers trained without its 92 g; the depth student needs it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import CollisionCfg

DASH_XML: Path = Path(__file__).parent / "xmls" / "dash_v1.xml"

DASH_JOINT_NAMES: tuple[str, ...] = (
    "l_hip_yaw",
    "l_hip_roll",
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "r_hip_yaw",
    "r_hip_roll",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
)


# Intel RealSense D435 on a printed jig at the bottom of the torso's front face (mjlab_dash
# dash_constants.py, commit db83fd6). Pose from the Onshape "Dash" document, Assembly 1 (frame
# matches the torso body frame to ~2 mm): the jig's seat faces 50 deg below horizontal and the
# camera's back sits flush on it. The realsense body's frame has +x along the optical axis,
# +y = torso +y, +z = away from the torso; both meshes are in that frame. Mass: 72 g (Intel's
# D435 spec) + ~20 g of jig. The depth stream at 848x480 (and 424x240, same aspect) spans
# 89.9 x 58.9 deg, from this unit's factory calibration (serial 825312071982).
REALSENSE_POS = (0.10607, 0.0, 0.03069)
REALSENSE_QUAT = (0.90631, 0.0, 0.42261, 0.0)
REALSENSE_HALF_SIZE = (0.0121, 0.0454, 0.0124)
REALSENSE_MASS = 0.072 + 0.020
REALSENSE_MESH_DIR = Path(__file__).parent / "xmls" / "assets_sensors"
REALSENSE_FOVY = 58.9
REALSENSE_CAMERA = "realsense_depth"


def _add_realsense(spec: mujoco.MjSpec) -> None:
    """Camera + jig as a body welded to the torso, with a depth camera at the lens.

    The box collider (*_collision) collides with the ground and the legs; the forearm is
    excluded from the torso subtree, so it gets an explicit pair with each forearm hull.
    """
    for name in ("realsense_d435", "realsense_jig"):
        spec.add_mesh(name=name, file=str(REALSENSE_MESH_DIR / f"{name}.stl"))
    hx, hy, hz = REALSENSE_HALF_SIZE
    m = REALSENSE_MASS
    body = spec.body("torso").add_body(name="realsense", pos=REALSENSE_POS, quat=REALSENSE_QUAT, mass=m,
                                       inertia=[m / 3 * (hy**2 + hz**2), m / 3 * (hx**2 + hz**2), m / 3 * (hx**2 + hy**2)])
    # Group 1, not the robot's visual group 2: drawn by the viewer but left out of the depth
    # render, since the camera sits on its own front glass and a mount perturbation would put
    # it inside the housing.
    for name in ("realsense_d435", "realsense_jig"):
        body.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, group=1, contype=0, conaffinity=0, density=0.0)
    body.add_geom(name="realsense_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(REALSENSE_HALF_SIZE), group=3, density=0.0)
    # MuJoCo cameras look along their -z. The image is upright (image up = +z, right = robot
    # right), as the raw D435 stream with the camera mounted upright.
    body.add_camera(name=REALSENSE_CAMERA, pos=[hx, 0.0, 0.0], xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0], fovy=REALSENSE_FOVY)
    for side in ("l", "r"):
        spec.add_pair(geomname1=f"{side}_lower_arm_hull", geomname2="realsense_collision", condim=1)


def get_spec(xml: Path = DASH_XML, realsense: bool = False) -> mujoco.MjSpec:
    """Dash spec with forearm-thigh contact restored via mesh hulls.

    dash_v1.xml excludes lower_arm/upper_leg because their capsules report false overlap
    through the walking arm swing. Mocap motions can still drive the hand into the thigh,
    which hardware then hits, so a hull of each visual mesh is added with an explicit pair
    between them. Pairs bypass the exclude and the contype/conaffinity masks, so the hulls
    touch nothing else.
    """
    spec = mujoco.MjSpec.from_file(str(xml))
    # The URDF import may carry a placeholder joint-level actuatorfrcrange, which MuJoCo
    # applies on top of the actuator forcerange; clear it so effort_limit rules alone.
    for joint in spec.joints:
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for side in ("l", "r"):
        for body in (f"{side}_lower_arm", f"{side}_upper_leg"):
            visual = spec.geom(f"{body}_visual")
            spec.body(body).add_geom(
                name=f"{body}_hull",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=visual.meshname,
                pos=visual.pos.tolist(),
                quat=visual.quat.tolist(),
                group=3,
                contype=0,
                conaffinity=0,
                density=0,
            )
        spec.add_pair(geomname1=f"{side}_lower_arm_hull", geomname2=f"{side}_upper_leg_hull", condim=1)
    if realsense:
        _add_realsense(spec)
    return spec


# WARNING: armature and the arm effort limit are ESTIMATES (scaled from robot mass, not motor
# datasheets); see mjlab_dash's dash_constants.py. Leg limits are hardware peak torques at the
# raised motor current (2026-11-10; were 50 / 80 / 35 estimates). Grouped by physical motor,
# not by limb: hip yaw/roll/pitch and shoulder_pitch share one motor, the knee has its own
# higher-torque one, the ankle and the other arm joints their own.
EFFORT_LIMIT_HIP = 80.0
EFFORT_LIMIT_KNEE = 120.0
EFFORT_LIMIT_ANKLE = 45.0
EFFORT_LIMIT_ARM = 30.0
ARMATURE = 0.01

# Command delay in 5 ms physics steps (0-25 ms, resampled every step) used by the
# mjlab_dash policies that ran on hardware.
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5

# (target joints, stiffness, damping, effort limit). Gains follow the motor grouping:
# 80/2 on every leg joint AND shoulder_pitch, 40/1 only on shoulder roll/yaw and elbow.
_ACTUATOR_GROUPS = (
    ((".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_shoulder_pitch"), 80.0, 2.0, EFFORT_LIMIT_HIP),
    ((".*_knee_pitch",), 80.0, 2.0, EFFORT_LIMIT_KNEE),
    ((".*_ankle_pitch",), 80.0, 2.0, EFFORT_LIMIT_ANKLE),
    ((".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow_pitch"), 40.0, 1.0, EFFORT_LIMIT_ARM),
)

# Root height measured so the soles sit on the ground at this pose.
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.6735),
    joint_pos={
        ".*_hip_pitch": -0.349,
        ".*_knee_pitch": 0.698,
        ".*_ankle_pitch": -0.349,
        ".*_elbow_pitch": -0.8,
        ".*_shoulder_pitch": 0.0,
    },
    joint_vel={".*": 0.0},
)

FOOT_GEOM_REGEX = r"^[lr]_foot_collision$"

# Everything collides. Feet get condim=3 and friction priority; every other collider is
# frictionless (condim=1) so grazing self-contacts don't fight the motion.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim={FOOT_GEOM_REGEX: 3, ".*_collision": 1},
    priority={FOOT_GEOM_REGEX: 1, ".*": 0},
    friction={FOOT_GEOM_REGEX: (0.6,)},
)

# Rad of joint target per unit of policy output. Set directly (not 0.25 * effort / kp as
# for mjlab's robots) because Dash's effort limits are estimates.
DASH_ACTION_SCALE: dict[str, float] = {".*": 0.25}


def get_dash_robot_cfg(command_delay: bool = False, realsense: bool = False) -> EntityCfg:
    """Fresh Dash robot config. ``command_delay`` enables the 0-25 ms actuator delay, ``realsense``
    adds the torso depth camera (camera ``robot/realsense_depth``)."""
    max_lag = DELAY_MAX_LAG if command_delay else 0
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            armature=ARMATURE,
            delay_min_lag=DELAY_MIN_LAG,
            delay_max_lag=max_lag,
        )
        for names, kp, kd, effort in _ACTUATOR_GROUPS
    )
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=functools.partial(get_spec, DASH_XML, realsense),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    )


# ----------------------------------------------------------------------------- hardware DR
# Ported from mjlab_dash's tasks/velocity/config/dash/hardware_dr.py (2026-10-09 hardware
# runs). Of these, only the torso COM offset enters xi; mass, PD gains, armature and joint
# friction are unobserved by the world model.

# Torso COM offset ranges (m). x is skewed rearward: standing ankle torque on the robot
# matches this model with the COM 2-3.5 cm in -x. mjlab_dash's latest tracking task pins x
# at -0.06 (commit d819b37); this is the walking task's range.
DASH_COM_RANGES: dict[int, tuple[float, float]] = {0: (-0.06, 0.06), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}


def add_hardware_randomization(events: dict[str, EventTermCfg], com_ranges: dict[int, tuple[float, float]] | None = None) -> None:
    """Add Dash's hardware-derived randomization to ``events`` (in place).

    Expects the ``base_com`` startup event on the torso. ``pseudo_inertia`` rebuilds
    body_ipos from the defaults, so it must run before ``base_com`` or it would erase the
    COM offset: ``base_com`` is moved after it.
    """
    events["base_com"].params["ranges"] = dict(com_ranges or DASH_COM_RANGES)
    # Mass: the robot weighs ~38 kg, the model 40.0. Per-body scale 0.88-1.05 (inertia too).
    events["body_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "alpha_range": (-0.064, 0.024)},
    )
    events["base_com"] = events.pop("base_com")
    # What the motor delivers per unit of kp/kd has never been measured.
    events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.7, 1.3),
            "operation": "scale",
        },
    )
    # Reflected rotor inertia is unmeasured; it grows with gear ratio squared, so the
    # ranges span that scaling widely.
    events["armature"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ranges": {
                r".*_hip_.*": (0.01, 0.03),
                r".*_shoulder_pitch": (0.01, 0.03),
                r".*_knee_pitch": (0.02, 0.05),
                r".*_ankle_pitch": (0.01, 0.03),
                r".*_(shoulder_roll|shoulder_yaw|elbow_pitch)": (0.008, 0.01),
            },
            "operation": "abs",
        },
    )
    # Coulomb friction at the output, from dash_home's sweep torque at steady speed.
    events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ranges": {
                r".*_hip_.*": (0.0, 2.5),
                r".*_knee_pitch": (0.0, 1.5),
                r".*_ankle_pitch": (0.0, 1.5),
                r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow_pitch)": (0.0, 2.0),
            },
            "operation": "abs",
        },
    )
