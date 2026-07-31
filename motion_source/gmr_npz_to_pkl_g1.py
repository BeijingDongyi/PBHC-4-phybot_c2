#!/usr/bin/env python3
"""Convert GMR retarget output (.npz) to the PBHC motion pkl format for Unitree G1 (23dof lock-wrist).

这是 gmr_npz_to_pkl.py 的 G1 版本。原 phybot 脚本一行没动, 两者可以共存:
    phybot_c2 -> gmr_npz_to_pkl.py       (21 dof, pose_aa (T,25,3))
    unitree_g1 -> 本脚本                  (23 dof, pose_aa (T,27,3))

与 phybot 版的三处实质差异:
  1. GMR 用的是 g1_mocap_29dof.xml, 输出 npz 的 dof_pos 是 **29 维**;
     PBHC 的 g1_23dof_lock_wrist 只要 23 维, 需要丢掉 6 个腕关节。
     裁切方式 = np.concatenate([dof[:, :19], dof[:, 22:26]], axis=1),
     与 count_pkl_contact_mask.py 里 count_pose_aa 的做法逐位一致:
        29dof idx  0..18  -> 23dof idx  0..18   (双腿12 + 腰3 + 左臂4)
        29dof idx 22..25  -> 23dof idx 19..22   (右臂4)
        丢弃 19,20,21 (左腕 roll/pitch/yaw) 与 26,27,28 (右腕 roll/pitch/yaw)
  2. STANCE 换成 g1_23dof_lock_wrist.yaml 的 default_joint_angles。
  3. pose_aa 尾部补 3 个零 (extend_config: left_hand_link / right_hand_link / head_link),
     G1 num_bodies=24, 所以 pose_aa = 1(root) + 23(dof) + 3(extend) = 27。

--limit-margin-frac 的作用与 phybot 版相同, 原因也相同: GMR 的 mink IK 会把 qpos 精确压在
XML 硬限位上, 参考轨迹贴着限位跑会让 limits_dof_pos 罚分永远躲不掉。

用法:
    python motion_source/gmr_npz_to_pkl_g1.py --npz ../GMR_phybot/data/output_npz/wushu_g1/kong_g1.npz
    # 之后跑 count_pkl_contact_mask.py 生成带 contact_mask 的版本

注意: 输入 npz 若已经是 23 维(比如上游改过), 脚本会跳过裁切并给出提示。
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(SCRIPT_DIR, "../description/robots/g1")

# g1_23dof_lock_wrist.yaml 的 init_state.default_joint_angles, canonical order
STANCE = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # right leg
    0.0, 0.0, 0.0,                            # waist yaw/roll/pitch
    0.2, 0.2, 0.0, 0.9,                       # left arm
    0.2, -0.2, 0.0, 0.9,                      # right arm
])

# 29dof -> 23dof: 丢掉两侧腕部各 3 个自由度
WRIST_KEEP = list(range(0, 19)) + list(range(22, 26))
N_DOF = 23
N_EXTEND = 3          # left_hand_link / right_hand_link / head_link


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


def slice_29_to_23(dof, name):
    """GMR 输出的 29 维 -> PBHC lock-wrist 的 23 维。已经是 23 维则原样返回。"""
    if dof.shape[1] == N_DOF:
        print(f"  {name}: dof 已经是 {N_DOF} 维, 跳过腕关节裁切")
        return dof
    assert dof.shape[1] == 29, (
        f"{name}: expected 29 (GMR g1_mocap_29dof) or {N_DOF} dof, got {dof.shape[1]}")
    print(f"  {name}: 29 -> {N_DOF} 维, 丢弃腕关节 idx 19,20,21,26,27,28")
    return dof[:, WRIST_KEEP]


def convert(npz_path, out_dir, dof_axis, joint_names, lo, hi, fps, margin_frac):
    name = os.path.splitext(os.path.basename(npz_path))[0]
    d = np.load(npz_path)
    root_trans = d["root_trans"].astype(np.float32)          # (T,3)
    root_rot = d["root_ori"].astype(np.float32)              # (T,4) xyzw
    dof = d["dof_pos"].astype(np.float32)                    # (T,29) 或 (T,23)

    dof = slice_29_to_23(dof, name)
    T = dof.shape[0]

    root_rot /= np.linalg.norm(root_rot, axis=1, keepdims=True)

    # soft-clamp away from URDF hard limits
    span = hi - lo
    lo_m, hi_m = lo + margin_frac * span, hi - margin_frac * span
    clamped = np.clip(dof, lo_m, hi_m).astype(np.float32)
    n_clamped = (np.abs(clamped - dof) > 1e-9).sum(axis=0)
    max_delta = np.abs(clamped - dof).max(axis=0)
    for i, n in enumerate(joint_names):
        if n_clamped[i]:
            print(f"  clamp {n:28s}: {n_clamped[i]:4d}/{T} frames, max delta {max_delta[i]:.4f} rad")
    dof = clamped

    root_aa = sRot.from_quat(root_rot).as_rotvec()[:, None, :]           # (T,1,3)
    pose_aa = np.concatenate(
        (root_aa, dof_axis * dof[:, :, None], np.zeros((T, N_EXTEND, 3))), axis=1
    ).astype(np.float32)                                                  # (T,27,3)
    assert pose_aa.shape[1] == 1 + N_DOF + N_EXTEND, pose_aa.shape

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
    if stance_diff >= 0.02:
        # phybot 版这里是硬 assert。G1 的 default_joint_angles 与 GMR 的收尾插值姿态未必一致
        # (GMR 只保证插到它自己的 default_dof), 所以先给警告而不是直接中断。
        print(f"  [WARN] {name}: 首帧与 G1 default stance 差 {stance_diff:.4f} rad (>0.02)。"
              f"检查 GMR 的收尾插值目标是否是 G1 的 default_joint_angles; "
              f"若差得多, 训练起步会有一个突跳。")
    print(f"  wrote {out_dir}/{name}.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True, help=".npz file or a directory of them")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "g1_pkl"))
    ap.add_argument("--fps", type=int, default=50, help="GMR bvh_to_robot_g1_kongti outputs 50 fps")
    ap.add_argument("--limit-margin-frac", type=float, default=0.05,
                    help="keep dofs this fraction of the joint range away from URDF limits")
    ap.add_argument("--urdf", default=os.path.join(ROBOT_DIR, "g1_23dof_lock_wrist.urdf"))
    ap.add_argument("--dof-axis", default=os.path.join(ROBOT_DIR, "dof_axis.npy"))
    args = ap.parse_args()

    dof_axis = np.load(args.dof_axis, allow_pickle=True).astype(np.float32)   # (23,3) g1
    assert dof_axis.shape == (N_DOF, 3), f"dof_axis 应为 ({N_DOF},3), 实际 {dof_axis.shape}"
    joint_names, lo, hi = load_urdf_limits(args.urdf)
    assert len(joint_names) == N_DOF, f"urdf revolute 关节数 {len(joint_names)} != {N_DOF}"

    files = sorted(glob.glob(os.path.join(args.npz, "*.npz"))) if os.path.isdir(args.npz) else [args.npz]
    assert files, f"no npz found at {args.npz}"

    os.makedirs(args.out_dir, exist_ok=True)
    for f in files:
        print(f"converting {f} (margin {args.limit_margin_frac})")
        convert(f, args.out_dir, dof_axis, joint_names, lo, hi, args.fps, args.limit_margin_frac)


if __name__ == "__main__":
    main()
