"""
导出 local_ref_rigid_body_pos_motionframe 的部署参考表。

用途:
    actor_obs 里新增的 local_ref_rigid_body_pos_motionframe(75 维)= 各刚体参考位置
    (25 个: 22 body + 3 extend, 锚在运动起点) 旋到"机器人当前航向"系。其中:
      - 旋转前的参考位置 P_ref[k] 是 phase 的确定函数 -> 离线预存成表
      - 航向旋转依赖机器人实时 yaw -> 部署端在线做
    本脚本把 P_ref[k] (k=0..N-1, 对应 motion_time=(k+1)*dt) 以及运动初始航向 phi0
    dump 成 C++ 可读的文本, 供 RL_deploy_kongti 加载。

复现的训练端代码: humanoidverse/envs/motion_tracking/motion_tracking.py:736-738
    motionframe_ref = ref_body_pos_extend - env_origins      # = rg_pos_t (offset=0)
    term = my_quat_rotate(heading_inv_rot, motionframe_ref)   # 在线部分, 不在本表里

运行(必须在 PBHC-main 根目录下, 因为 motion asset 是相对路径):
    python humanoidverse/export_ref_motionframe.py \
        --config logs/phybot_c2/kongti/0717_1139/config.yaml \
        --out    ../PhybotSoftware_c2/RL_deploy_kongti/data/kongti_ref_motionframe.txt

输出文本格式:
    第 1 行: num_frames  dim  dt  motion_len  motion_init_yaw
    其后 num_frames 行: 每行 dim(=75) 个浮点, 即 P_ref[k] 展平 (body0.xyz, body1.xyz, ...)
"""
import argparse
import math
import os

import torch
from omegaconf import OmegaConf

# 注册 ${eval:...} 等自定义 resolver (与 eval_agent 一致); 失败不影响已解析的 config.yaml
try:
    from humanoidverse.utils.config_utils import *  # noqa: F401,F403
except Exception:
    pass


def _yaw_from_quat_xyzw(q: torch.Tensor) -> float:
    """与 isaac_utils.calc_heading 同约定: 把机体 x 轴旋到世界系, atan2(y, x)。q = [x,y,z,w]。"""
    x, y, z, w = [float(v) for v in q]
    # R * [1,0,0]
    rx = 1.0 - 2.0 * (y * y + z * z)
    ry = 2.0 * (x * y + w * z)
    return math.atan2(ry, rx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="训练 run 的 config.yaml (含已解析的 robot.motion)")
    parser.add_argument("--out", required=True, help="输出参考表 txt 路径")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="策略步长, 与训练 200fps/decim4 = 0.02 一致")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    motion_cfg = config.robot.motion

    from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
    motion_lib = MotionLibRobotWJX(motion_cfg, num_envs=1, device=args.device)
    motion_lib.load_motions(random_sample=False)

    motion_ids = torch.zeros(1, dtype=torch.long, device=args.device)
    motion_len = float(motion_lib.get_motion_length(motion_ids).item())
    num_frames = int(math.ceil(motion_len / args.dt))

    # phi0: 机器人进入 kick 时的参考航向 = 运动 t=0 的 root yaw (部署端以此为 yaw 零点对齐)
    res0 = motion_lib.get_motion_state(motion_ids, torch.zeros(1, device=args.device))
    phi0 = _yaw_from_quat_xyzw(res0["root_rot"][0])

    rows = []
    for k in range(num_frames):
        # 与训练一致: motion_time = (episode_length + 1) * dt, 且 offset=None -> 纯运动帧
        t = torch.full((1,), (k + 1) * args.dt, device=args.device)
        res = motion_lib.get_motion_state(motion_ids, t)  # offset 默认 None
        p_ref = res["rg_pos_t"][0]  # (num_extended_bodies, 3), phybot = (25, 3)
        rows.append(p_ref.reshape(-1).cpu())

    table = torch.stack(rows, dim=0)  # (num_frames, dim)
    dim = table.shape[1]
    assert dim == 3 * (config.robot.num_bodies + motion_cfg.nums_extend_bodies), \
        f"dim={dim} 与 3*(num_bodies+extend) 不符, 检查 robot 配置"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"{num_frames} {dim} {args.dt:.6f} {motion_len:.6f} {phi0:.8f}\n")
        for k in range(num_frames):
            f.write(" ".join(f"{v:.8f}" for v in table[k].tolist()) + "\n")

    # sanity: root(z) 峰值应体现跳跃 (base_link 是 body 0, z 是第 3 个分量)
    root_z = table[:, 2]
    print(f"[export_ref_motionframe] num_frames={num_frames} dim={dim} "
          f"motion_len={motion_len:.3f}s dt={args.dt}")
    print(f"[export_ref_motionframe] motion_init_yaw phi0={phi0:.4f} rad "
          f"({math.degrees(phi0):.2f} deg)")
    print(f"[export_ref_motionframe] ref root z: min={root_z.min():.3f} "
          f"max={root_z.max():.3f} (跳跃时应明显抬高)")
    print(f"[export_ref_motionframe] written -> {args.out}")


if __name__ == "__main__":
    main()
