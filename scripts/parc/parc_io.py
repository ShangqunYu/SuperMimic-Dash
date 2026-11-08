"""PARC motion-terrain files: load, forward kinematics (numpy), terrain height lookup."""
import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np

HUMANOID_XML = os.environ.get("PARC_HUMANOID_XML", "/media/shangqunyu/Dash/parc_src/PARC/data/assets/humanoid.xml")  # github.com/mshoe/PARC


def load(f):
    c = pickle.load(open(f, "rb"))
    m = pickle.loads(c["motion_data"])
    t = pickle.loads(c["terrain_data"]) if c.get("terrain_data") else None
    return m, t


def skeleton(xml=HUMANOID_XML):
    names, parents, offs = [], [], []
    def rec(node, parent):
        i = len(names)
        names.append(node.get("name")); parents.append(parent)
        offs.append(np.fromstring(node.get("pos", "0 0 0"), sep=" "))
        for ch in node.findall("body"):
            rec(ch, i)
    rec(ET.parse(xml).getroot().find("worldbody").find("body"), -1)
    return names, np.array(parents), np.array(offs)


def qmul(a, b):  # xyzw
    ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):  # xyzw
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def fk(m, sk=None):
    names, parents, offs = sk or skeleton()
    T = len(m["root_pos"])
    pos = np.zeros((T, len(names), 3)); rot = np.zeros((T, len(names), 4))
    pos[:, 0], rot[:, 0] = m["root_pos"], m["root_rot"]
    for j in range(1, len(names)):
        p = parents[j]
        pos[:, j] = pos[:, p] + qrot(rot[:, p], np.broadcast_to(offs[j], (T, 3)))
        rot[:, j] = qmul(rot[:, p], m["joint_rot"][:, j - 1])
    return names, pos, rot


def hf_height(t, xy):
    """Height of the blocky terrain at xy (cells of size dx centred on min_point + i*dx)."""
    ij = np.round((np.asarray(xy) - t["min_point"]) / t["dx"]).astype(int)
    i = np.clip(ij[..., 0], 0, t["hf"].shape[0] - 1); j = np.clip(ij[..., 1], 0, t["hf"].shape[1] - 1)
    return t["hf"][i, j]
