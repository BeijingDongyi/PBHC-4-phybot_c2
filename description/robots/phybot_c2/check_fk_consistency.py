#!/usr/bin/env python3
"""Phase-1 verification gate: phybot_fitmotion.xml must agree body-for-body with
the training URDF (phybot_c2.urdf).

Both models are loaded in MuJoCo and compared with its own FK -- fully
independent of both build_phybot_fitmotion.py's math and Humanoid_Batch:

  1. same 21 hinge joints, identical local axis and range per joint;
  2. at zero pose and N random in-range poses, world position AND orientation
     of all 22 common bodies agree to <1e-6;
  3. the URDF's fixed links (torso/neck_yaw/neck_pitch), which MuJoCo keeps as
     welded bodies, sit exactly at waist_yaw (+) the URDF fixed-joint offsets --
     i.e. the geometry the fitmotion file folded away is where it should be.

The URDF is preprocessed in-memory (visual/collision elements dropped) so its
mesh path style (package:// vs meshes/) does not matter here.

Usage: python3 check_fk_consistency.py [--n 100] [--seed 0]
"""

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "phybot_c2.urdf")
FITMOTION_PATH = os.path.join(SCRIPT_DIR, "phybot_fitmotion.xml")
TOL = 1e-6


def load_stripped_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    # MuJoCo's URDF importer defaults to fusestatic=true, which would fuse the
    # joint-less base_link/torso/neck bodies away -- exactly what we must inspect.
    mj_ext = ET.SubElement(root, "mujoco")
    ET.SubElement(mj_ext, "compiler", fusestatic="false", balanceinertia="true")
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        tree.write(f, encoding="unicode")
        tmp = f.name
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


def joint_map(model):
    out = {}
    for jid in range(model.njnt):
        name = mujoco.mj_name2id  # noqa: silence linters; use id2name below
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            out[name] = jid
    return out


def body_xpos_xmat(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name} not found"
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def rot_angle(r1, r2):
    cos = (np.trace(r1.T @ r2) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def fixed_chain_offsets(urdf_path, anchor="waist_yaw"):
    """{fixed_link: offset in anchor frame} accumulated over the URDF fixed joints."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    offsets, frontier = {}, [(anchor, np.zeros(3))]
    while frontier:
        parent, base = frontier.pop()
        for child, j in joints.items():
            if j.find("parent").get("link") != parent or j.get("type") != "fixed":
                continue
            o = j.find("origin")
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            assert np.allclose(rpy, 0), f"fixed joint {j.get('name')} has rpy, extend this check"
            xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            offsets[child] = base + xyz
            frontier.append((child, offsets[child]))
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of random poses")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m_urdf = load_stripped_urdf(URDF_PATH)
    m_fit = mujoco.MjModel.from_xml_path(FITMOTION_PATH)
    d_urdf, d_fit = mujoco.MjData(m_urdf), mujoco.MjData(m_fit)

    j_urdf, j_fit = joint_map(m_urdf), joint_map(m_fit)
    assert set(j_urdf) == set(j_fit), \
        f"joint sets differ: {set(j_urdf) ^ set(j_fit)}"
    names = sorted(j_urdf)
    print(f"joints: {len(names)} common hinge joints")

    for n in names:
        ax_u, ax_f = m_urdf.jnt_axis[j_urdf[n]], m_fit.jnt_axis[j_fit[n]]
        rg_u, rg_f = m_urdf.jnt_range[j_urdf[n]], m_fit.jnt_range[j_fit[n]]
        assert np.allclose(ax_u, ax_f, atol=1e-9), f"{n}: axis {ax_u} vs {ax_f}"
        assert np.allclose(rg_u, rg_f, atol=1e-9), f"{n}: range {rg_u} vs {rg_f}"
    print("per-joint axis + range: identical")

    bodies = ["base_link"] + [
        mujoco.mj_id2name(m_fit, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(1, m_fit.nbody)
        if mujoco.mj_id2name(m_fit, mujoco.mjtObj.mjOBJ_BODY, b) != "base_link"
    ]
    assert len(bodies) == 22, f"expected 22 fitmotion bodies, got {len(bodies)}"

    fixed_offsets = fixed_chain_offsets(URDF_PATH)
    assert set(fixed_offsets) == {"torso", "neck_yaw", "neck_pitch"}, fixed_offsets

    rng = np.random.default_rng(args.seed)
    lo = np.array([m_urdf.jnt_range[j_urdf[n]][0] for n in names])
    hi = np.array([m_urdf.jnt_range[j_urdf[n]][1] for n in names])

    max_pos_err = max_rot_err = max_fold_err = 0.0
    worst = ("", -1)
    for trial in range(args.n + 1):
        q = np.zeros(len(names)) if trial == 0 else rng.uniform(lo, hi)
        for n, v in zip(names, q):
            d_urdf.qpos[m_urdf.jnt_qposadr[j_urdf[n]]] = v
            d_fit.qpos[m_fit.jnt_qposadr[j_fit[n]]] = v
        d_fit.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]  # free joint -> match URDF's welded base
        mujoco.mj_forward(m_urdf, d_urdf)
        mujoco.mj_forward(m_fit, d_fit)

        for b in bodies:
            p_u, r_u = body_xpos_xmat(m_urdf, d_urdf, b)
            p_f, r_f = body_xpos_xmat(m_fit, d_fit, b)
            e = float(np.linalg.norm(p_u - p_f))
            if e > max_pos_err:
                max_pos_err, worst = e, (b, trial)
            max_rot_err = max(max_rot_err, rot_angle(r_u, r_f))

        p_w, r_w = body_xpos_xmat(m_urdf, d_urdf, "waist_yaw")
        for link, off in fixed_offsets.items():
            p_l, _ = body_xpos_xmat(m_urdf, d_urdf, link)
            max_fold_err = max(max_fold_err, float(np.linalg.norm(p_l - (p_w + r_w @ off))))

    print(f"poses tested: zero + {args.n} random (seed {args.seed})")
    print(f"max body position error:    {max_pos_err:.3e} m   (worst: {worst[0]}, trial {worst[1]})")
    print(f"max body orientation error: {max_rot_err:.3e} rad")
    print(f"max folded-link error:      {max_fold_err:.3e} m   (torso/neck vs waist_yaw+offset)")

    ok = max_pos_err < TOL and max_rot_err < TOL and max_fold_err < TOL
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
