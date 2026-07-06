#!/usr/bin/env python3
"""Convert GMR retarget output (.npz) to the PBHC motion pkl format for phybot_c2.

GMR (GMR_phybot/scripts/bvh_to_robot_c2.py) emits 50 fps npz with:
    root_trans (T,3), root_ori (T,4 xyzw), dof_pos (T,21 canonical order), ...
The first/last ~30 frames are already interpolated to the default stance by GMR,
so do NOT stack PBHC's own interpolation on top.

Output schema matches convert_lafan_pkl.py: {name: {root_trans_offset, pose_aa
(T,25,3 = root + 21 joints + 3 extend zeros), dof, root_rot, smpl_joints, fps}}.
Run count_pkl_contact_mask.py robot=phybot_c2 on the output folder afterwards.

--limit-margin-frac keeps every dof at least frac*(upper-lower) away from the
URDF hard limits. The margin must be applied HERE (not upstream): GMR's mink IK
pins qpos exactly onto the XML hard limits (kong has elbows on 0.0 and knees on
-0.05), and a reference that rides the limits forces a permanent
limits_dof_pos penalty the policy cannot avoid.

Usage:
    python gmr_npz_to_pkl.py --npz /path/to/kong.npz            # single file
    python gmr_npz_to_pkl.py --npz /path/to/npz_dir             # every *.npz
"""

import argparse
import glob
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(SCRIPT_DIR, "../description/robots/phybot_c2")

STANCE = np.array([
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
    0.0,
    0.0, 0.1, 0.0, -0.1,
    0.0, -0.1, 0.0, -0.1,
])  # default_joint_angles, canonical order (== phybot_21dof.yaml)


def load_urdf_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    lims = {}
    for j in root.findall("joint"):
        if j.get("type") == "revolute":
            lim = j.find("limit")
            lims[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    order = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    lo = np.array([lims[n][0] for n in order])
    hi = np.array([lims[n][1] for n in order])
    return order, lo, hi


def convert(npz_path, out_dir, dof_axis, joint_names, lo, hi, fps, margin_frac):
    name = os.path.splitext(os.path.basename(npz_path))[0]
    d = np.load(npz_path)
    root_trans = d["root_trans"].astype(np.float32)          # (T,3)
    root_rot = d["root_ori"].astype(np.float32)              # (T,4) xyzw
    dof = d["dof_pos"].astype(np.float32)                    # (T,21)
    T = dof.shape[0]
    assert dof.shape[1] == 21, f"{name}: expected 21 dof, got {dof.shape[1]}"

    root_rot /= np.linalg.norm(root_rot, axis=1, keepdims=True)

    # soft-clamp away from URDF hard limits
    span = hi - lo
    lo_m, hi_m = lo + margin_frac * span, hi - margin_frac * span
    clamped = np.clip(dof, lo_m, hi_m).astype(np.float32)
    n_clamped = (np.abs(clamped - dof) > 1e-9).sum(axis=0)
    max_delta = np.abs(clamped - dof).max(axis=0)
    for i, n in enumerate(joint_names):
        if n_clamped[i]:
            print(f"  clamp {n:22s}: {n_clamped[i]:4d}/{T} frames, max delta {max_delta[i]:.4f} rad")
    dof = clamped

    root_aa = sRot.from_quat(root_rot).as_rotvec()[:, None, :]           # (T,1,3)
    pose_aa = np.concatenate(
        (root_aa, dof_axis * dof[:, :, None], np.zeros((T, 3, 3))), axis=1
    ).astype(np.float32)                                                  # (T,25,3)

    joblib.dump(
        {name: {
            "root_trans_offset": root_trans,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_rot,
            "smpl_joints": pose_aa,
            "fps": fps,
        }},
        os.path.join(out_dir, f"{name}.pkl"),
    )

    stance_diff = np.abs(dof[0] - np.clip(STANCE, lo_m, hi_m)).max()
    print(f"  {name}: T={T} ({T / fps:.2f}s @ {fps}fps)  root z [{root_trans[:, 2].min():.3f}, "
          f"{root_trans[:, 2].max():.3f}]  frame0-vs-stance {stance_diff:.4f}")
    assert stance_diff < 0.02, f"{name}: frame 0 is not the default stance -- wrong motion or dof order?"
    print(f"  wrote {out_dir}/{name}.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True, help=".npz file or a directory of them")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "phybot_pkl"))
    ap.add_argument("--fps", type=int, default=50, help="GMR bvh_to_robot_c2 outputs 50 fps")
    ap.add_argument("--limit-margin-frac", type=float, default=0.05,
                    help="keep dofs this fraction of the joint range away from URDF limits")
    ap.add_argument("--urdf", default=os.path.join(ROBOT_DIR, "phybot_c2.urdf"))
    ap.add_argument("--dof-axis", default=os.path.join(ROBOT_DIR, "dof_axis.npy"))
    args = ap.parse_args()

    dof_axis = np.load(args.dof_axis).astype(np.float32)     # (21,3) phybot, NOT the g1 copy
    joint_names, lo, hi = load_urdf_limits(args.urdf)
    files = sorted(glob.glob(os.path.join(args.npz, "*.npz"))) if os.path.isdir(args.npz) else [args.npz]
    assert files, f"no npz found at {args.npz}"

    os.makedirs(args.out_dir, exist_ok=True)
    for f in files:
        print(f"converting {f} (margin {args.limit_margin_frac})")
        convert(f, args.out_dir, dof_axis, joint_names, lo, hi, args.fps, args.limit_margin_frac)


if __name__ == "__main__":
    main()
