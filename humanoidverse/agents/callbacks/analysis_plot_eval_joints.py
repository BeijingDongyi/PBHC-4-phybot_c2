"""eval 跑够 n_runs 遍动作后, 把最后一遍完整画出来, 布局对齐
PhybotSoftware_c2/tools/plot_deploy_log.py: 每行一个物理量, 每列一个关节,
最后一行是机体状态。和部署日志图放一起就能逐格对比 sim 与实机。

为什么要等几遍: 第一遍常常摔, 摔了会自动 reset 重来, 等到第 n 遍再画就基本是
一次干净的动作。每遍的时长和是否跑满 motion 会打在标题和终端里, 万一第 n 遍也摔了
一眼就能看出来。

注意 evaluate_policy 是 while True, 永远走不到 on_post_evaluate_policy, 所以出图
只能在 on_post_eval_env_step 里按 episode 计数触发, 不能挂在"跑完"上。
"""
import os
import time

import numpy as np

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback

BLUE = "#005A9D"

# 每行一个物理量: (buffer key, 行标题, 单位)
ROWS = [
    ("dof_pos", "dof_pos", "rad"),
    ("dof_vel", "dof_vel", "rad/s"),
    ("action", "action", "-"),
    ("torque", "torque", "N·m"),
    ("pos_err", "pos_err", "rad"),
]
BASE_ITEMS = [("roll", "deg"), ("pitch", "deg"), ("yaw", "deg"),
              ("wx", "rad/s"), ("wy", "rad/s"), ("wz", "rad/s"), ("phase", "-")]


class AnalysisPlotEvalJoints(RL_EvalCallback):
    # 类属性兜底: 关掉时 on_pre_evaluate_policy 会提前返回, 这两个不会被实例赋值,
    # 但 on_post_eval_env_step 每步都要读, 没有兜底就是 AttributeError。
    enabled = False
    plotted = False

    def __init__(self, config, training_loop):
        super().__init__(config, training_loop)
        self.env = training_loop.env
        self.enabled = False
        if not self.config.get("enable", True):
            return  # base_eval.yaml 里默认关着, 用 plot_joints=True 打开

        self.env_idx = self.config.get("env_idx", 0)
        self.n_runs = int(self.config.get("n_runs", 3))
        self.save_png = self.config.get("save_png", True)
        self.enabled = True

    # ------------------------------------------------------------------ setup

    def on_pre_evaluate_policy(self):
        if not self.enabled:
            return
        env = self.env
        dof_names = list(env.config.robot.dof_names)

        sel = self.config.get("joints", None)
        if sel:  # 支持只画一部分关节
            self.cols = [dof_names.index(n) if isinstance(n, str) else int(n) for n in sel]
        else:
            self.cols = list(range(len(dof_names)))
        self.names = [dof_names[i] for i in self.cols]

        self.runs_done = 0
        self.run_summary = []
        self.plotted = False
        self._reset_buffers()

        # motion 跑满需要多少步, 用来判断这一遍是跑完了还是中途摔了
        self.full_steps = None
        if hasattr(env, "motion_len"):
            self.full_steps = float(env.motion_len[self.env_idx]) / env.dt

    def _reset_buffers(self):
        self.buf = {k: [] for k, _, _ in ROWS}
        self.base = {k: [] for k, _ in BASE_ITEMS}

    # ------------------------------------------------------------------ 采样

    def on_post_eval_env_step(self, actor_state):
        if not self.enabled:
            return actor_state
        if self.plotted:
            if actor_state["step"] % 20 == 0:
                self.fig.canvas.flush_events()  # 图已经出了, 只保持窗口能拖动
            return actor_state

        env, i, cols = self.env, self.env_idx, self.cols

        def np_(x):
            return x.detach().cpu().numpy()

        dof_pos = np_(env.simulator.dof_pos[i])[cols]
        ref = np_(env.ref_joint_pos[i])[cols] if hasattr(env, "ref_joint_pos") \
            else np.full(len(cols), np.nan)
        self.buf["dof_pos"].append(dof_pos)
        self.buf["dof_vel"].append(np_(env.simulator.dof_vel[i])[cols])
        self.buf["action"].append(np_(env.actions[i])[cols])
        self.buf["torque"].append(np_(env.torques[i])[cols])
        self.buf["pos_err"].append(dof_pos - ref)

        from humanoidverse.utils.torch_utils import get_euler_xyz
        r, p, y = get_euler_xyz(env.base_quat[i:i + 1])
        wrap = lambda v: np.degrees(float(v) - 2 * np.pi * (float(v) > np.pi))
        self.base["roll"].append(wrap(r))
        self.base["pitch"].append(wrap(p))
        self.base["yaw"].append(wrap(y))
        for k, v in zip(("wx", "wy", "wz"), np_(env.base_ang_vel[i])):
            self.base[k].append(float(v))
        phase = float(env.episode_length_buf[i]) * env.dt / float(env.motion_len[i]) \
            if hasattr(env, "motion_len") else np.nan
        self.base["phase"].append(min(phase, 1.05))

        if bool(actor_state["dones"][i]):
            self.runs_done += 1
            n = len(self.buf["dof_pos"])
            full = self.full_steps is None or n >= 0.95 * self.full_steps
            self.run_summary.append((n, full))
            from loguru import logger
            logger.info(f"[plot_joints] 第 {self.runs_done}/{self.n_runs} 遍结束: "
                        f"{n} 步 ({n * env.dt:.2f}s), "
                        f"{'跑满' if full else '中途终止(可能摔了)'}")
            if self.runs_done >= self.n_runs:
                self._plot()
                self.plotted = True
            else:
                self._reset_buffers()
        return actor_state

    # ------------------------------------------------------------------ 出图

    def _plot(self):
        import matplotlib
        headless = matplotlib.get_backend().lower().startswith(("agg", "template"))
        import matplotlib.pyplot as plt
        from loguru import logger

        data = {k: np.asarray(v) for k, v in self.buf.items()}
        base = {k: np.asarray(v) for k, v in self.base.items()}
        t = np.arange(len(data["dof_pos"])) * self.env.dt
        n = len(self.cols)

        figsize = self.config.get("figsize", [min(2.0 * n, 44), 2.1 * (len(ROWS) + 1)])
        fig, axs = plt.subplots(len(ROWS) + 1, n, figsize=tuple(figsize),
                                sharex=True, squeeze=False)

        for r, (key, title, unit) in enumerate(ROWS):
            for c in range(n):
                a = axs[r][c]
                a.plot(t, data[key][:, c], color=BLUE, lw=0.8)
                a.grid(alpha=0.3)
                a.tick_params(labelsize=6)
                if r == 0:
                    a.set_title(self.names[c], fontsize=7)
                if c == 0:
                    a.set_ylabel(f"{title}\n[{unit}]", fontsize=8)
                # y 轴交给 matplotlib 自适应实际数据。不画关节限位: 限位比实测大一个量级时
                # (比如力矩限 90 而实测只有十几) 会把曲线压成一条直线, 什么都看不出来。

        rb = len(ROWS)
        for c in range(n):
            a = axs[rb][c]
            if c >= len(BASE_ITEMS):
                a.axis("off")
                continue
            key, unit = BASE_ITEMS[c]
            a.plot(t, np.unwrap(base[key], period=360) if key == "yaw" else base[key],
                   color=BLUE, lw=0.8)
            a.grid(alpha=0.3)
            a.tick_params(labelsize=6)
            a.set_title(f"{key} [{unit}]", fontsize=7)
            a.set_xlabel("t [s]", fontsize=7)

        steps, full = self.run_summary[-1]
        fig.suptitle(f"eval joints - run {self.runs_done}/{self.n_runs}, "
                     f"{steps} steps ({steps * self.env.dt:.2f}s), "
                     f"{'full motion' if full else 'TERMINATED EARLY (likely fell)'}",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self.fig = fig

        if self.save_png:
            out_dir = self.config.get("save_dir", None) or \
                os.path.join(str(self.env.config.get("ckpt_dir", ".")), "eval_plots")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"joints_{time.strftime('%m%d_%H%M%S')}.png")
            fig.savefig(path, dpi=100)
            logger.info(f"[plot_joints] 已保存 {path}")

        if not headless:
            plt.show(block=bool(self.config.get("block", False)))
            fig.canvas.flush_events()
