#!/usr/bin/env python3
"""Build phybot_fitmotion.xml and dof_axis.npy from the official phybot_c2 URDF.

The output MJCF is the kinematics-only skeleton consumed by Humanoid_Batch
(humanoidverse/utils/motion_lib/torch_humanoid_batch.py and its copies under
motion_source/ and smpl_retarget/). It is NOT used for physics simulation --
the simulator loads phybot_c2.urdf directly.

Humanoid_Batch's parsing assumptions, which this file must satisfy:
  * one root body carrying a single free joint, then exactly one hinge joint
    per descendant body (dof_pos is extracted as pose[..., 1:num_bodies]);
  * every hinge joint has an integer axis and an explicit range;
  * a non-empty <actuator> block whose motor names match the joint names;
  * <compiler meshdir> pointing at the STL directory (used by load_mesh()).

Body frames must coincide with the URDF joint frames that IsaacGym keeps
after collapsing fixed joints, because motion_tracking.py subtracts
ref/sim body positions index by index. Hence every fixed URDF joint
(torso, neck_yaw, neck_pitch) is folded into its CHILDREN's offsets.
Folding a fixed joint into the parent body's own position is what GMR's
phybot_c2.xml did (waist_yaw placed at torso height z=0.253 instead of
0.146) and it shifts that body's reference position permanently.

Usage:
    python3 build_phybot_fitmotion.py [--urdf PATH] [--out-dir PATH]

By default the URDF is read from this script's directory (the copy made in
migration step 1.1). The official source of truth is
PhybotSoftware_c2/RobotModel/phybot_c2/urdf/phybot_c2.urdf.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Canonical 21-DoF order shared by GMR npz, the robot yaml and the deploy
# stack: left leg 6, right leg 6, waist_yaw, left arm 4, right arm 4.
CANONICAL_DOF_ORDER = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch",
]

ROOT_LINK = "base_link"
ROOT_STANDING_Z = 0.85  # cosmetic only: FK takes root translation from motion data


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx  # URDF rpy is extrinsic X-Y-Z


def matrix_to_quat_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
        w, x, y, z = q
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fmt(values, nd=8):
    out = []
    for v in np.atleast_1d(np.asarray(values, dtype=float)):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for l in root.findall("link"):
        entry = {"name": l.get("name"), "mesh": None, "inertial": None}
        vis = l.find("visual")
        if vis is not None:
            mesh = vis.find("geometry/mesh")
            if mesh is not None:
                entry["mesh"] = os.path.basename(mesh.get("filename"))
            vo = vis.find("origin")
            if vo is not None:
                xyz = np.fromstring(vo.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(vo.get("rpy", "0 0 0"), sep=" ")
                assert np.allclose(xyz, 0) and np.allclose(rpy, 0), \
                    f"non-identity visual origin on {entry['name']}: extend the generator"
        inertial = l.find("inertial")
        if inertial is not None:
            io = inertial.find("origin")
            ixyz = np.fromstring(io.get("xyz", "0 0 0"), sep=" ") if io is not None else np.zeros(3)
            irpy = np.fromstring(io.get("rpy", "0 0 0"), sep=" ") if io is not None else np.zeros(3)
            inz = inertial.find("inertia")
            entry["inertial"] = {
                "pos": ixyz,
                "rpy": irpy,
                "mass": float(inertial.find("mass").get("value")),
                "fullinertia": [float(inz.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")],
            }
        links[entry["name"]] = entry

    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        entry = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "rot": rpy_to_matrix(rpy),
        }
        if entry["type"] in ("revolute", "continuous"):
            axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            assert all(a in (-1.0, 0.0, 1.0) for a in axis), \
                f"non-integer axis on {entry['name']}: {axis} (Humanoid_Batch casts axis to int)"
            limit = j.find("limit")
            assert limit is not None and limit.get("lower") and limit.get("upper"), \
                f"missing limit on {entry['name']} (from_mjcf requires an explicit range)"
            entry["axis"] = axis.astype(int)
            entry["range"] = (float(limit.get("lower")), float(limit.get("upper")))
        joints.append(entry)

    return links, joints


def fold_fixed_joints(links, joints):
    """Return, per movable (revolute) joint: its nearest movable ancestor link and
    the accumulated transform from that ancestor's frame, folding fixed joints.
    Also collect the visuals of fixed links, re-anchored on their movable ancestor.
    """
    joint_by_child = {j["child"]: j for j in joints}

    def accumulate(joint):
        """Transform of `joint`'s frame expressed in nearest movable ancestor frame."""
        pos, rot = joint["xyz"].copy(), joint["rot"].copy()
        parent = joint_by_child.get(joint["parent"])
        while parent is not None and parent["type"] == "fixed":
            pos = parent["xyz"] + parent["rot"] @ pos
            rot = parent["rot"] @ rot
            parent = joint_by_child.get(parent["parent"])
        anchor = joint["parent"]
        while anchor != ROOT_LINK and joint_by_child[anchor]["type"] == "fixed":
            anchor = joint_by_child[anchor]["parent"]
        return anchor, pos, rot

    movable = {}
    for j in joints:
        if j["type"] in ("revolute", "continuous"):
            anchor, pos, rot = accumulate(j)
            movable[j["child"]] = {"joint": j, "anchor": anchor, "pos": pos, "rot": rot}

    folded_visuals = {}  # movable ancestor link -> [(mesh, pos, rot)]
    for j in joints:
        if j["type"] != "fixed":
            continue
        mesh = links[j["child"]]["mesh"]
        if mesh is None:
            continue
        anchor, pos, rot = accumulate(j)
        folded_visuals.setdefault(anchor, []).append((mesh, pos, rot))

    return movable, folded_visuals


def build_mjcf(links, joints, movable, folded_visuals):
    children = {}  # movable link -> ordered movable child links (canonical order)
    for name in CANONICAL_DOF_ORDER:
        info = movable[name]
        children.setdefault(info["anchor"], []).append(name)

    mj = ET.Element("mujoco", model="phybot_c2_fitmotion")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes",
                  autolimits="true", balanceinertia="true")
    asset = ET.SubElement(mj, "asset")
    used_meshes = []

    def add_mesh(mesh_file):
        name = os.path.splitext(mesh_file)[0]
        if name not in used_meshes:
            used_meshes.append(name)
            ET.SubElement(asset, "mesh", name=name, file=mesh_file)
        return name

    def add_inertial(body_el, link_name):
        ine = links[link_name]["inertial"]
        attrs = {"pos": fmt(ine["pos"]), "mass": fmt([ine["mass"]]),
                 "fullinertia": fmt(ine["fullinertia"], nd=10)}
        if not np.allclose(ine["rpy"], 0):
            attrs["quat"] = fmt(matrix_to_quat_wxyz(rpy_to_matrix(ine["rpy"])))
        ET.SubElement(body_el, "inertial", **attrs)

    def add_visual_geom(body_el, mesh_file, pos=None, rot=None):
        attrs = {"type": "mesh", "contype": "0", "conaffinity": "0",
                 "group": "1", "density": "0", "mesh": add_mesh(mesh_file)}
        if pos is not None and not np.allclose(pos, 0):
            attrs["pos"] = fmt(pos)
        if rot is not None and not np.allclose(rot, np.eye(3)):
            attrs["quat"] = fmt(matrix_to_quat_wxyz(rot))
        ET.SubElement(body_el, "geom", **attrs)

    def add_body(parent_el, link_name):
        info = movable[link_name]
        attrs = {"name": link_name, "pos": fmt(info["pos"])}
        if not np.allclose(info["rot"], np.eye(3)):
            attrs["quat"] = fmt(matrix_to_quat_wxyz(info["rot"]))
        body = ET.SubElement(parent_el, "body", **attrs)
        add_inertial(body, link_name)
        j = info["joint"]
        ET.SubElement(body, "joint", name=j["name"], pos="0 0 0",
                      axis=" ".join(str(a) for a in j["axis"]),
                      range=fmt(j["range"]))
        if links[link_name]["mesh"]:
            add_visual_geom(body, links[link_name]["mesh"])
        for mesh, pos, rot in folded_visuals.get(link_name, []):
            add_visual_geom(body, mesh, pos, rot)
        for child in children.get(link_name, []):
            add_body(body, child)

    worldbody = ET.SubElement(mj, "worldbody")
    root_body = ET.SubElement(worldbody, "body", name=ROOT_LINK,
                              pos=fmt([0, 0, ROOT_STANDING_Z]))
    add_inertial(root_body, ROOT_LINK)
    ET.SubElement(root_body, "joint", name="floating_base_joint", type="free",
                  limited="false", actuatorfrclimited="false")
    if links[ROOT_LINK]["mesh"]:
        add_visual_geom(root_body, links[ROOT_LINK]["mesh"])
    for mesh, pos, rot in folded_visuals.get(ROOT_LINK, []):
        add_visual_geom(root_body, mesh, pos, rot)
    for child in children.get(ROOT_LINK, []):
        add_body(root_body, child)

    actuator = ET.SubElement(mj, "actuator")
    for name in CANONICAL_DOF_ORDER:
        jname = movable[name]["joint"]["name"]
        ET.SubElement(actuator, "motor", name=jname, joint=jname)

    return mj


def zero_pose_world_positions_from_urdf(joints):
    """World position of every link frame at zero pose, walking the full URDF
    chain including fixed joints (independent of the folding code above)."""
    world = {ROOT_LINK: (np.zeros(3), np.eye(3))}
    remaining = [j for j in joints if j["type"] in ("revolute", "continuous", "fixed")]
    while remaining:
        progressed = False
        for j in list(remaining):
            if j["parent"] in world:
                ppos, prot = world[j["parent"]]
                world[j["child"]] = (ppos + prot @ j["xyz"], prot @ j["rot"])
                remaining.remove(j)
                progressed = True
        assert progressed, f"disconnected joints: {[j['name'] for j in remaining]}"
    return {k: v[0] for k, v in world.items()}


def zero_pose_world_positions_from_mjcf(mj_root):
    world = {}

    def walk(body_el, ppos, prot):
        pos = np.fromstring(body_el.get("pos", "0 0 0"), sep=" ")
        quat = np.fromstring(body_el.get("quat", "1 0 0 0"), sep=" ")
        w, x, y, z = quat
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        wpos = ppos + prot @ pos
        wrot = prot @ rot
        world[body_el.get("name")] = wpos
        for c in body_el.findall("body"):
            walk(c, wpos, wrot)

    root = mj_root.find("worldbody").find("body")
    walk(root, -np.fromstring(root.get("pos"), sep=" "), np.eye(3))  # cancel cosmetic root z
    world[ROOT_LINK] = np.zeros(3)
    return world


def indent_tree(elem, level=0):  # ET.indent needs py3.9+; this box may be older
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            indent_tree(child, level + 1)
            if not (child.tail or "").strip():
                child.tail = pad + "  "
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=os.path.join(SCRIPT_DIR, "phybot_c2.urdf"))
    ap.add_argument("--out-dir", default=SCRIPT_DIR)
    args = ap.parse_args()

    links, joints = parse_urdf(args.urdf)
    urdf_dof_order = [j["name"] for j in joints if j["type"] in ("revolute", "continuous")]
    assert urdf_dof_order == CANONICAL_DOF_ORDER, \
        f"URDF joint order changed!\n  urdf: {urdf_dof_order}\n  expected: {CANONICAL_DOF_ORDER}"

    movable, folded_visuals = fold_fixed_joints(links, joints)
    mj = build_mjcf(links, joints, movable, folded_visuals)

    # -- self check: zero-pose world positions, URDF full chain vs generated tree --
    ref = zero_pose_world_positions_from_urdf(joints)
    got = zero_pose_world_positions_from_mjcf(mj)
    print(f"{'body':24s} {'urdf world (x y z)':28s} err")
    max_err = 0.0
    for name in [ROOT_LINK] + CANONICAL_DOF_ORDER:
        err = float(np.linalg.norm(ref[name] - got[name]))
        max_err = max(max_err, err)
        print(f"{name:24s} {fmt(ref[name], 6):28s} {err:.2e}")
    # XML attributes are rounded to 8 decimals, so ~1e-9 round-trip error is expected
    assert max_err < 1e-7, f"FK mismatch between URDF and generated MJCF: {max_err}"
    for name, z in (("waist_yaw", 0.146), ("left_shoulder_pitch", 0.366), ("right_shoulder_pitch", 0.366)):
        assert abs(ref[name][2] - z) < 1e-9, f"{name} expected z={z}, got {ref[name][2]}"
    print(f"\nself-check OK, max |ref-gen| = {max_err:.2e}")
    print(f"folded fixed-link visuals: "
          f"{ {k: [m for m, _, _ in v] for k, v in folded_visuals.items()} }")

    indent_tree(mj)
    xml_path = os.path.join(args.out_dir, "phybot_fitmotion.xml")
    ET.ElementTree(mj).write(xml_path, encoding="unicode")
    print(f"wrote {xml_path}")

    dof_axis = np.array([movable[n]["joint"]["axis"] for n in CANONICAL_DOF_ORDER], dtype=np.int64)
    axis_path = os.path.join(args.out_dir, "dof_axis.npy")
    np.save(axis_path, dof_axis)
    print(f"wrote {axis_path}  shape={dof_axis.shape} dtype={dof_axis.dtype}")


if __name__ == "__main__":
    main()
