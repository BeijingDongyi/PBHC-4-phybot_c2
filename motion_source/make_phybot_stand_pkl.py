#!/usr/bin/env python3
"""Generate a synthetic standing motion pkl for phybot_c2 smoke tests.

Produces motion_source/phybot_pkl_stand/stand.pkl: T frames of the default
stance (== phybot_21dof.yaml default_joint_angles == deploy stance) with the
root fixed at z=0.75, identity orientation, 50 fps. Same schema as
convert_lafan_pkl.py output; run count_pkl_contact_mask.py on the folder
afterwards to add the contact mask:

    python make_phybot_stand_pkl.py
    python count_pkl_contact_mask.py robot=phybot_c2 +input_folder=phybot_pkl_stand

Intended use: end-to-end pipeline checks (Humanoid_Batch FK on the phybot
assets) and training dry-runs before real motion data exists (phase 4).
"""

import os

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

T = 100          # frames
FPS = 50         # GMR convention
ROOT_Z = 0.75    # standing root height (phybot_mimic init_state)

# canonical 21-dof order: left leg 6, right leg 6, waist_yaw, left arm 4, right arm 4
STANCE = np.array([
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # left leg: hip_pitch/roll/yaw, knee, ankle_pitch/roll
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # right leg
    0.0,                              # waist_yaw
    0.0, 0.1, 0.0, -0.1,              # left arm: shoulder_pitch/roll/yaw, elbow_pitch
    0.0, -0.1, 0.0, -0.1,             # right arm
], dtype=np.float32)


def main():
    dof = np.tile(STANCE, (T, 1))                                   # (T, 21)
    root_trans = np.tile([0.0, 0.0, ROOT_Z], (T, 1)).astype(np.float32)
    root_rot = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)  # xyzw identity

    dof_axis = np.load(
        os.path.join(SCRIPT_DIR, "../description/robots/phybot_c2/dof_axis.npy")
    ).astype(np.float32)                                            # (21, 3)

    root_aa = np.zeros((T, 1, 3), dtype=np.float32)                 # identity rotvec
    pose_aa = np.concatenate(
        (root_aa, dof_axis * dof[:, :, None], np.zeros((T, 3, 3), dtype=np.float32)),
        axis=1,
    )                                                               # (T, 25, 3)

    out_dir = os.path.join(SCRIPT_DIR, "phybot_pkl_stand")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        {"stand": {
            "root_trans_offset": root_trans,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_rot,
            "smpl_joints": pose_aa,
            "fps": FPS,
        }},
        os.path.join(out_dir, "stand.pkl"),
    )
    print(f"wrote {out_dir}/stand.pkl  (T={T}, fps={FPS}, root_z={ROOT_Z})")


if __name__ == "__main__":
    main()
