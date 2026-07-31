#!/usr/bin/env python3
"""Edit a pre-contact-mask motion pkl: re-ground floating contact frames,
time-retime a segment, and/or apply a smooth local joint correction.

Operates on the SAME schema as gmr_npz_to_pkl.py's output (root_trans_offset,
pose_aa, dof, root_rot, fps -- no contact_mask/real smpl_joints yet). Run
count_pkl_contact_mask.py on the output folder afterwards to regenerate the
contact mask and real FK positions.

--ground: shifts root_trans_offset z so that on contact-labeled frames the
ankle_roll joints (median, from an existing *_cont_mask.pkl of the SAME
motion) sit at the URDF resting height ANKLE_REST_Z, i.e. the sole touches
z=0 exactly. foot_detect() only checks height < 0.12m and low velocity, so a
retargeted foot can be marked "in contact" while still floating a few cm.

--retime start:end:factor (repeatable): linearly time-stretches the given
frame range (end exclusive) by factor (>1 slows down / adds frames, <1 speeds
up). dof and root_trans_offset are resampled with linear interpolation,
root_rot with quaternion slerp. pose_aa is rebuilt from the new dof/root_rot
(not interpolated directly) using dof_axis, same convention as
gmr_npz_to_pkl.py. Frames outside all segments are left untouched (1:1).

--joint-bias joint:center:half_width:amplitude (repeatable): adds a
raised-cosine bias to one joint after retiming. The correction is zero at
center +/- half_width and reaches amplitude at center, so both endpoints join
the original trajectory with zero slope.

Usage:
    python edit_motion.py --pkl phybot_pkl/kong.pkl \\
        --ground phybot_pkl_contact_mask/kong_cont_mask.pkl \\
        --retime 55:80:2.0 \\
        --joint-bias 2:193:10:-0.1 \\
        --out phybot_pkl/kong_slow_grounded.pkl
"""

import argparse
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot, Slerp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(SCRIPT_DIR, "../description/robots/phybot_c2")


# Resting height of the ankle_roll joint origin when the foot sits flat on the
# floor. From phybot_c2.urdf: the foot collision is two x-axis cylinder rails,
# origin z=-0.04, radius 0.012 -> lowest point z = -0.052 below the joint.
# smpl_joints idx 6/12 (foot_detect's fid) ARE the ankle_roll joint origins,
# so a correctly grounded reference has them at +0.052 on contact frames,
# NOT at 0 -- shifting them to 0 buries the foot 5.2cm underground.
ANKLE_REST_Z = 0.052


def compute_ground_bias(cont_mask_pkl, target=ANKLE_REST_Z):
    d = joblib.load(cont_mask_pkl)
    m = d[list(d.keys())[0]]
    cm = m["contact_mask"]
    smpl = m["smpl_joints"]
    fid_l, fid_r = 6, 12  # ankle_roll joints, matches count_pkl_contact_mask.py foot_detect()
    l_on, r_on = cm[:, 0] == 1, cm[:, 1] == 1
    pooled = np.concatenate([smpl[l_on, fid_l, 2], smpl[r_on, fid_r, 2]])
    bias = float(np.median(pooled)) - target
    print(f"  ground bias from {cont_mask_pkl}: median ankle z {np.median(pooled):.4f} - rest {target:.4f} = shift {bias:+.4f}m")
    return bias


def retime_segment(dof, root_trans, root_rot, start, end, factor):
    n_orig = end - start
    n_new = max(2, round(n_orig * factor))
    t_src = np.arange(n_orig)
    t_dst = np.linspace(0, n_orig - 1, n_new)

    dof_seg = np.stack([np.interp(t_dst, t_src, dof[start:end, j]) for j in range(dof.shape[1])], axis=1)
    trans_seg = np.stack([np.interp(t_dst, t_src, root_trans[start:end, k]) for k in range(3)], axis=1)

    slerp = Slerp(t_src, sRot.from_quat(root_rot[start:end]))
    rot_seg = slerp(t_dst).as_quat()

    return dof_seg.astype(np.float32), trans_seg.astype(np.float32), rot_seg.astype(np.float32)


def apply_retime(dof, root_trans, root_rot, segments):
    segments = sorted(segments, key=lambda s: s[0])
    out_dof, out_trans, out_rot = [], [], []
    cursor = 0
    for start, end, factor in segments:
        assert start >= cursor, f"segments must be sorted and non-overlapping (cursor={cursor}, got start={start})"
        out_dof.append(dof[cursor:start]); out_trans.append(root_trans[cursor:start]); out_rot.append(root_rot[cursor:start])
        d, t, r = retime_segment(dof, root_trans, root_rot, start, end, factor)
        print(f"  retime frames [{start}:{end}) x{factor} -> {end - start} frames become {len(d)} frames")
        out_dof.append(d); out_trans.append(t); out_rot.append(r)
        cursor = end
    out_dof.append(dof[cursor:]); out_trans.append(root_trans[cursor:]); out_rot.append(root_rot[cursor:])
    return np.concatenate(out_dof), np.concatenate(out_trans), np.concatenate(out_rot)


def apply_joint_biases(dof, biases):
    """Apply smooth raised-cosine corrections in output-frame space."""
    result = dof.copy()
    frame_ids = np.arange(len(result))
    for joint, center, half_width, amplitude in biases:
        if not 0 <= joint < result.shape[1]:
            raise ValueError(
                f"joint index {joint} is outside [0, {result.shape[1]})"
            )
        if half_width <= 0:
            raise ValueError("joint-bias half_width must be positive")
        if not 0 <= center < len(result):
            raise ValueError(
                f"joint-bias center {center} is outside [0, {len(result)})"
            )

        weights = np.zeros(len(result), dtype=np.float64)
        active = np.abs(frame_ids - center) <= half_width
        weights[active] = 0.5 * (
            1.0
            + np.cos(
                np.pi * (frame_ids[active] - center) / half_width
            )
        )
        result[:, joint] += amplitude * weights
        print(
            f"  joint bias dof[{joint}] center={center} "
            f"half_width={half_width} amplitude={amplitude:+.4f} rad"
        )
    return result.astype(np.float32)


def rebuild_pose_aa(dof, root_rot, dof_axis):
    root_aa = sRot.from_quat(root_rot).as_rotvec()[:, None, :]
    T = dof.shape[0]
    return np.concatenate(
        (root_aa, dof_axis * dof[:, :, None], np.zeros((T, 3, 3), dtype=np.float32)), axis=1
    ).astype(np.float32)


def parse_segment(s):
    start, end, factor = s.split(":")
    return int(start), int(end), float(factor)


def parse_joint_bias(s):
    joint, center, half_width, amplitude = s.split(":")
    return int(joint), int(center), int(half_width), float(amplitude)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", required=True, help="pre-contact-mask input pkl (gmr_npz_to_pkl.py output)")
    ap.add_argument("--ground", default=None, help="existing *_cont_mask.pkl of the SAME motion, used to compute the height bias")
    ap.add_argument("--ground-target", type=float, default=ANKLE_REST_Z, help="desired ankle_roll z on contact frames (URDF sole geometry)")
    ap.add_argument("--retime", action="append", default=[], help="start:end:factor, repeatable, sorted & non-overlapping")
    ap.add_argument(
        "--joint-bias",
        action="append",
        default=[],
        help=(
            "joint:center:half_width:amplitude, repeatable; applied after "
            "retiming in output-frame space"
        ),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--dof-axis", default=os.path.join(ROBOT_DIR, "dof_axis.npy"))
    args = ap.parse_args()

    d = joblib.load(args.pkl)
    m = d[list(d.keys())[0]]
    name = os.path.splitext(os.path.basename(args.out))[0]
    dof, root_trans, root_rot, fps = m["dof"], m["root_trans_offset"], m["root_rot"], m["fps"]

    if args.ground:
        bias = compute_ground_bias(args.ground, args.ground_target)
        root_trans = root_trans.copy()
        root_trans[:, 2] -= bias

    if args.retime:
        segments = [parse_segment(s) for s in args.retime]
        dof, root_trans, root_rot = apply_retime(dof, root_trans, root_rot, segments)

    if args.joint_bias:
        dof = apply_joint_biases(
            dof, [parse_joint_bias(s) for s in args.joint_bias]
        )

    dof_axis = np.load(args.dof_axis).astype(np.float32)
    pose_aa = rebuild_pose_aa(dof, root_rot, dof_axis)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({name: {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": pose_aa,  # placeholder; count_pkl_contact_mask.py overwrites via real FK
        "fps": fps,
    }}, args.out)
    print(f"wrote {args.out}  T={dof.shape[0]} ({dof.shape[0] / fps:.2f}s @ {fps}fps)")


if __name__ == "__main__":
    main()
