"""温度归一化 + PDE 无量纲化版 SA-PINN 复现脚本（TensorFlow 2.15 + WSL2 Ubuntu 22.04）。

放置位置：/home/qm/work/PINNs_Adaptive_thermal/main_nondim.py

核心改进（对比 main_repro.py）：
1. 坐标无量纲化：x_nd = x / L, y_nd = y / L, t_nd = t / t_ref ∈ [0, 1]，
   并在网络输入前线性映射到 [-1, 1]，适配 tanh 激活函数。
2. 温度归一化（仅基于训练集统计，无测试集泄露）：
   T_nd = (T - T_ref) / delta_T ∈ [0, 1]，预测后反归一化回开尔文温度 T (K)。
3. 控制方程无量纲化：
   d(T_nd)/d(t_nd) - Fo * (d2(T_nd)/d(x_nd)^2 + d2(T_nd)/d(y_nd)^2) - Q_nd = 0
   其中傅里叶数 Fo = k * t_ref / (rho * cp * L^2) ≈ 0.2445，
   无量纲热源 Q_nd = (t_ref / (rho * cp * delta_T)) * Q(L*x_nd, L*y_nd, t_ref*t_nd)。
4. 稳健 PDE 残差损失：默认使用 log(1 + (w * f_nd)^2)（对应原仓库 README 思路），
   防止极小光斑中心处的 2D/3D 奇异残差淹没数据与边界项（可通过 --pde-loss mse 切换）。
5. 固定样本-权重绑定：先一次性打乱并固定数据/配点批次，使 SA-PINN 点级权重在跨 epoch 时真正绑定同一批物理点。
6. 完整定量评估与可信绘图：左右两图共享同一色条量程，输出 RMSE (K)、相对 L2 误差 (%)、
   峰值温度/峰值坐标对比，并在训练结束自动保存 loss_curve.png。
"""

import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import netCDF4
import numpy as np
from scipy import interpolate
from scipy.stats import qmc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import wandb


class DataPrep:
    """读取 MOOSE Exodus II (.e) 文件并构建顶表面 (z=5) 的 3D 规则网格线性插值器。"""

    def __init__(self, xmax=1.0, filename="function_heat_source_out_100W_6s.e"):
        with netCDF4.Dataset(filename) as nc:
            temps = np.ma.getdata(nc.variables["vals_nod_var1"][:]).astype(np.float64)
        self.temp = temps.reshape((121, 6, 101, 101))
        x = np.linspace(0.0, xmax, 101)
        y = np.linspace(0.0, xmax, 101)
        t = np.linspace(0.0, 6.0, 121)
        self.temp_func = interpolate.RegularGridInterpolator(
            (t, x, y), self.temp[:, 5, :, :]
        )

    def get_coords(self, xmin, xmax, ymin, ymax, tmin, num_x, num_y, num_t):
        x = np.linspace(xmin, xmax, num=num_x)
        y = np.linspace(ymin, ymax, num=num_y)
        t = np.linspace(tmin, tmin + ((num_t - 1) * 0.05), num=num_t)
        xxx, yyy, ttt = np.meshgrid(x, y, t)
        return np.concatenate(
            (ttt.flatten()[:, None], xxx.flatten()[:, None], yyy.flatten()[:, None]),
            axis=1,
        )

    def get_temps(self, coords, temp_cutoff=20000.0):
        temp_vals = self.temp_func(coords).flatten()[:, None]
        idx = np.where(temp_vals <= temp_cutoff)[0]
        return np.concatenate((coords[idx], temp_vals[idx]), axis=1)


def build_neural_net(layer_sizes, dtype_str):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],), dtype=dtype_str))
    for width in layer_sizes[1:-1]:
        model.add(
            layers.Dense(
                width,
                activation=tf.nn.tanh,
                kernel_initializer="glorot_normal",
                dtype=dtype_str,
            )
        )
    model.add(
        layers.Dense(
            layer_sizes[-1],
            activation=None,
            kernel_initializer="glorot_normal",
            dtype=dtype_str,
        )
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalized & Non-Dimensionalized SA-PINN Thermal Model"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: CPU 友好快速收敛模式; full: 原仓库大网络全量规模",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="数值精度（默认 quick 用 float32 提速，full 用 float64）",
    )
    parser.add_argument(
        "--pde-loss",
        choices=["log", "mse"],
        default="log",
        help="PDE 残差损失形式：log=log(1+(w*f)^2) 更稳健，mse=(w*f)^2",
    )
    parser.add_argument(
        "--pde-weight",
        type=float,
        default=0.01,
        help="无量纲 PDE 残差项的权重系数（默认 0.01）",
    )
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=0.1,
        help="无量纲绝热边界项的权重系数（默认 0.1）",
    )
    parser.add_argument(
        "--lr-net",
        type=float,
        default=1e-3,
        help="网络参数 Adam 学习率（默认 1e-3）",
    )
    parser.add_argument(
        "--lr-weights",
        type=float,
        default=1e-3,
        help="自适应点权重 Adam 梯度上升学习率（默认 1e-3）",
    )
    parser.add_argument(
        "--hotspot-ratio",
        type=float,
        default=0.0,
        help="可选：数据采样中高温熔池区域的强化采样比例 (0.0~0.5，默认 0.0 为纯均匀采样)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--data-file",
        type=str,
        default="function_heat_source_out_100W_6s.e",
        help="MOOSE Exodus II (.e) 文件路径",
    )
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint 目录")
    parser.add_argument("--out-dir", type=str, default=None, help="结果图像输出目录")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="WandB 日志模式",
    )
    return parser.parse_args()


def sample_training_data(X_data_all, N_data, T_ref, delta_T, hotspot_ratio, rng):
    """从全部监督网格中抽取固定训练样本（支持纯均匀采样或高温区强化采样）。"""
    total = X_data_all.shape[0]
    N_data = min(N_data, total)
    if hotspot_ratio <= 0.0:
        idx = rng.choice(total, size=N_data, replace=False)
        return X_data_all[idx]

    hot_mask = X_data_all[:, 3] >= (T_ref + 0.15 * delta_T)
    hot_idx = np.where(hot_mask)[0]
    cold_idx = np.where(~hot_mask)[0]

    n_hot = min(int(N_data * hotspot_ratio), len(hot_idx))
    n_cold = N_data - n_hot
    sel_hot = rng.choice(hot_idx, size=n_hot, replace=False)
    sel_cold = rng.choice(cold_idx, size=n_cold, replace=False)
    sel = np.concatenate([sel_hot, sel_cold])
    rng.shuffle(sel)
    return X_data_all[sel]


def save_eval_plot(T_true, T_pred, epoch, rmse, rel_l2, out_path):
    """绘制左右共享色条量程的三联对比图（真值 | 预测 | 绝对误差）。"""
    im_true = np.reshape(T_true, (101, 101), order="F")
    im_pred = np.reshape(T_pred, (101, 101), order="F")
    im_err = np.abs(im_pred - im_true)

    vmin = float(im_true.min())
    vmax = float(im_true.max())

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 3.8), ncols=3)

    p1 = ax1.imshow(im_true, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")
    fig.colorbar(p1, ax=ax1, extend="both", label="Temperature (K)")
    ax1.set_title("MOOSE Reference (t=5.0s)")

    p2 = ax2.imshow(im_pred, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")
    fig.colorbar(p2, ax=ax2, extend="both", label="Temperature (K)")
    ax2.set_title(f"Non-Dim SA-PINN (Epoch {epoch})")

    p3 = ax3.imshow(im_err, cmap="jet", vmin=0.0, vmax=max(float(im_err.max()), 1.0), interpolation="none")
    fig.colorbar(p3, ax=ax3, extend="max", label="Abs Error (K)")
    ax3.set_title(f"Abs Error (RMSE={rmse:.1f} K, RelL2={rel_l2 * 100:.2f}%)")

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_loss_curve(history, out_path):
    """保存训练损失与测试集 RMSE / Rel-L2 收敛曲线。"""
    fig, (ax1, ax2) = plt.subplots(figsize=(11, 4.2), ncols=2)

    epochs = history["epoch"]
    ax1.semilogy(epochs, history["mse_data_nd"], label="Data MSE (nd)")
    ax1.semilogy(epochs, history["mse_0_nd"], label="IC MSE (nd)")
    ax1.semilogy(epochs, history["mse_b_nd"], label="BC MSE (nd)")
    ax1.semilogy(epochs, history["loss_colloc_nd"], label="PDE Loss (nd)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dimensionless Loss (log scale)")
    ax1.set_title("Training Loss Components")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history["eval_epoch"]:
        ax2.plot(history["eval_epoch"], history["eval_rmse"], "o-", color="tab:blue", label="Test RMSE (K)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test RMSE (K)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.grid(True, alpha=0.3)

        ax2_r = ax2.twinx()
        ax2_r.plot(
            history["eval_epoch"],
            [v * 100.0 for v in history["eval_rel_l2"]],
            "s--",
            color="tab:red",
            label="Rel L2 (%)",
        )
        ax2_r.set_ylabel("Relative L2 Error (%)", color="tab:red")
        ax2_r.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_title("Test Evaluation @ t=5.0s")

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    dtype_str = args.dtype or ("float32" if args.mode == "quick" else "float64")
    tf_dtype = tf.float32 if dtype_str == "float32" else tf.float64
    np_dtype = np.float32 if dtype_str == "float32" else np.float64
    tf.keras.backend.set_floatx(dtype_str)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.mode == "quick":
        layer_sizes = [3, 64, 64, 64, 64, 1]
        batch_size = 512
        N_data = 16384
        N_colloc = 16384
        N_initial = 1000
        N_boundary = 1000
        tf_iter = 30 if args.epochs is None else args.epochs
        plot_every = 5
        ckpt_dir = args.ckpt_dir or "./tf_ckpts_nondim_quick"
        out_dir = args.out_dir or "./outputs_nondim_quick"
    else:
        layer_sizes = [3, 128, 128, 128, 128, 128, 1]
        batch_size = 2048
        N_data = 256000
        N_colloc = 256000
        N_initial = 5000
        N_boundary = 5000
        tf_iter = 50 if args.epochs is None else args.epochs
        plot_every = 10
        ckpt_dir = args.ckpt_dir or "./tf_ckpts_nondim_full"
        out_dir = args.out_dir or "./outputs_nondim_full"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    num_batches = N_colloc // batch_size

    # 特征尺度（物理坐标 -> 无量纲坐标）
    L_ref = 1.0      # 空间特征长度（与原数据 x, y ∈ [0, 1] 对齐）
    t_ref = 6.0      # 时间特征尺度（覆盖 0~6s 全过程，使 t_nd ∈ [0, 1]）

    # 1. 加载数据并仅根据训练集统计温度归一化参数 (T_ref, delta_T)
    print(f"[Data] Loading {args.data_file} ...")
    t_load = time.time()
    temp_data = DataPrep(xmax=L_ref, filename=args.data_file)

    X_data_all = temp_data.get_temps(
        temp_data.get_coords(0.01, 0.99, 0.01, 0.99, 0.05, 99, 99, 96)
    )
    X_0_all = temp_data.get_temps(
        temp_data.get_coords(0.01, 0.99, 0.01, 0.99, 0.0, 99, 99, 1)
    )

    # 仅使用训练时间窗口 (t <= 4.80s) 计算归一化参数，避免测试集 (t=5.0s) 泄露
    T_train_min = float(min(X_data_all[:, 3].min(), X_0_all[:, 3].min()))
    T_train_max = float(max(X_data_all[:, 3].max(), X_0_all[:, 3].max()))
    T_ref = T_train_min
    delta_T = max(T_train_max - T_train_min, 1e-6)

    # 物理参数与无量纲数计算
    rho_val = 8e-3
    cp_val = 500e4
    k_val = 16.3e2
    r_val = 1e-2
    v_val = 1.8e-1
    q_peak_dim = 6.0 * math.sqrt(3.0 / (math.pi ** 3)) * 7e11

    Fo_val = (k_val * t_ref) / (rho_val * cp_val * (L_ref ** 2))
    S_char = (rho_val * cp_val * delta_T) / t_ref
    Q_nd_peak = q_peak_dim / S_char

    print(
        f"[Norm] T_ref={T_ref:.2f} K | T_max(train)={T_train_max:.2f} K | delta_T={delta_T:.2f} K\n"
        f"[PDE ] Fo={Fo_val:.6f} | S_char={S_char:.3e} | Q_nd_peak={Q_nd_peak:.3e} | "
        f"pde_loss={args.pde_loss} | pde_weight={args.pde_weight}"
    )
    print(
        f"[Conf] mode={args.mode} | dtype={dtype_str} | layers={layer_sizes} | epochs={tf_iter} | "
        f"N_data={N_data} | N_colloc={N_colloc} | batch_size={batch_size} ({num_batches} batches/epoch)"
    )
    print(f"[Dev ] {tf.config.list_physical_devices()}")

    # 2. 构建并一次性固定无量纲数据集（确保 SA-PINN 权重与物理点严格对应）
    X_data_sel = sample_training_data(
        X_data_all, N_data, T_ref, delta_T, args.hotspot_ratio, rng
    )
    X_data_nd = np.column_stack(
        [
            X_data_sel[:, 0] / t_ref,
            X_data_sel[:, 1] / L_ref,
            X_data_sel[:, 2] / L_ref,
            (X_data_sel[:, 3] - T_ref) / delta_T,
        ]
    ).astype(np_dtype)

    idx_0 = rng.choice(X_0_all.shape[0], size=min(N_initial, X_0_all.shape[0]), replace=False)
    X_0_sel = X_0_all[idx_0]
    t_0_nd = tf.constant((X_0_sel[:, 0:1] / t_ref).astype(np_dtype))
    x_0_nd = tf.constant((X_0_sel[:, 1:2] / L_ref).astype(np_dtype))
    y_0_nd = tf.constant((X_0_sel[:, 2:3] / L_ref).astype(np_dtype))
    T_0_nd = tf.constant(((X_0_sel[:, 3:4] - T_ref) / delta_T).astype(np_dtype))
    n_init_actual = X_0_sel.shape[0]

    # 拉丁超立方采样无量纲配点：t_nd ∈ [0, 4.85/6.0], x_nd ∈ [0, 1], y_nd ∈ [0, 1]
    lb_nd = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ub_nd = np.array([4.85 / t_ref, 1.0, 1.0], dtype=np.float64)
    lhs_engine = qmc.LatinHypercube(d=3, seed=args.seed)
    X_colloc_nd = qmc.scale(lhs_engine.random(n=N_colloc), lb_nd, ub_nd).astype(np_dtype)

    X_ul_all = temp_data.get_coords(0.01, 0.99, 0.0, 1.0, 0.05, 99, 2, 96)
    X_ul_sel = X_ul_all[rng.choice(X_ul_all.shape[0], size=N_boundary, replace=False)]
    t_ul_nd = tf.constant((X_ul_sel[:, 0:1] / t_ref).astype(np_dtype))
    x_ul_nd = tf.constant((X_ul_sel[:, 1:2] / L_ref).astype(np_dtype))
    y_ul_nd = tf.constant((X_ul_sel[:, 2:3] / L_ref).astype(np_dtype))

    X_lr_all = temp_data.get_coords(0.0, 1.0, 0.01, 0.99, 0.05, 2, 99, 96)
    X_lr_sel = X_lr_all[rng.choice(X_lr_all.shape[0], size=N_boundary, replace=False)]
    t_lr_nd = tf.constant((X_lr_sel[:, 0:1] / t_ref).astype(np_dtype))
    x_lr_nd = tf.constant((X_lr_sel[:, 1:2] / L_ref).astype(np_dtype))
    y_lr_nd = tf.constant((X_lr_sel[:, 2:3] / L_ref).astype(np_dtype))

    X_test = temp_data.get_temps(
        temp_data.get_coords(0.0, 1.0, 0.0, 1.0, 5.0, 101, 101, 1),
        temp_cutoff=50000.0,
    )
    t_test_nd = tf.constant((X_test[:, 0:1] / t_ref).astype(np_dtype))
    x_test_nd = tf.constant((X_test[:, 1:2] / L_ref).astype(np_dtype))
    y_test_nd = tf.constant((X_test[:, 2:3] / L_ref).astype(np_dtype))
    T_test_true = X_test[:, 3:4]

    # 固定分批（不再每个 epoch 打乱，确保 SA-PINN 权重与物理点一一对应）
    data_batches = [
        tf.constant(X_data_nd[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
    ]
    colloc_batches = [
        tf.constant(X_colloc_nd[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
    ]
    print(f"[Data] Ready in {time.time() - t_load:.2f}s.")

    wandb.init(project="sa-pinns-nondim", mode=args.wandb_mode)

    # 3. 构建网络与无量纲物理算子
    u_model = build_neural_net(layer_sizes, dtype_str)

    Fo = tf.constant(Fo_val, dtype=tf_dtype)
    Q_nd_coeff = tf.constant(Q_nd_peak, dtype=tf_dtype)
    L_c = tf.constant(L_ref, dtype=tf_dtype)
    t_c_ref = tf.constant(t_ref, dtype=tf_dtype)
    r_c = tf.constant(r_val, dtype=tf_dtype)
    v_c = tf.constant(v_val, dtype=tf_dtype)
    half_c = tf.constant(0.5, dtype=tf_dtype)
    neg3_c = tf.constant(-3.0, dtype=tf_dtype)
    two_c = tf.constant(2.0, dtype=tf_dtype)
    one_c = tf.constant(1.0, dtype=tf_dtype)
    pde_w_c = tf.constant(args.pde_weight, dtype=tf_dtype)
    bc_w_c = tf.constant(args.bc_weight, dtype=tf_dtype)
    use_log_pde = args.pde_loss == "log"

    def predict_nd(x_nd, y_nd, t_nd):
        # 将 [0, 1] 无量纲坐标线性映射到 [-1, 1] 送入 tanh 网络
        inp = two_c * tf.concat([x_nd, y_nd, t_nd], axis=1) - one_c
        return u_model(inp)

    @tf.function
    def f_model_nd(x_nd, y_nd, t_nd):
        T_nd = predict_nd(x_nd, y_nd, t_nd)
        T_x = tf.gradients(T_nd, x_nd)[0]
        T_y = tf.gradients(T_nd, y_nd)[0]
        T_xx = tf.gradients(T_x, x_nd)[0]
        T_yy = tf.gradients(T_y, y_nd)[0]
        T_t = tf.gradients(T_nd, t_nd)[0]

        fx = v_c * (t_c_ref * t_nd)
        c1 = (L_c * x_nd - fx) / r_c
        c2 = (L_c * y_nd - half_c) / r_c
        Q_nd = Q_nd_coeff * tf.math.exp(neg3_c * (c1 ** 2 + c2 ** 2))

        return T_t - Fo * (T_xx + T_yy) - Q_nd

    @tf.function
    def u_derv_y_nd(x_nd, y_nd, t_nd):
        T_nd = predict_nd(x_nd, y_nd, t_nd)
        return tf.gradients(T_nd, y_nd)[0]

    @tf.function
    def u_derv_x_nd(x_nd, y_nd, t_nd):
        T_nd = predict_nd(x_nd, y_nd, t_nd)
        return tf.gradients(T_nd, x_nd)[0]

    def compute_loss_nd(
        x_d, y_d, t_d, T_d,
        x_0, y_0, t_0, T_0,
        x_c, y_c, t_c,
        x_ul, y_ul, t_ul,
        x_lr, y_lr, t_lr,
        w_d, w_0, w_c, w_ul, w_lr,
    ):
        T_d_pred = predict_nd(x_d, y_d, t_d)
        mse_data_sa = tf.reduce_mean(tf.square(w_d * (T_d - T_d_pred)))
        mse_data = tf.reduce_mean(tf.square(T_d - T_d_pred))

        T_0_pred = predict_nd(x_0, y_0, t_0)
        mse_0_sa = tf.reduce_mean(tf.square(w_0 * (T_0 - T_0_pred)))
        mse_0 = tf.reduce_mean(tf.square(T_0 - T_0_pred))

        f_pred = f_model_nd(x_c, y_c, t_c)
        if use_log_pde:
            loss_colloc_sa = pde_w_c * tf.reduce_mean(tf.math.log1p(tf.square(w_c * f_pred)))
            loss_colloc_raw = tf.reduce_mean(tf.math.log1p(tf.square(f_pred)))
        else:
            loss_colloc_sa = pde_w_c * tf.reduce_mean(tf.square(w_c * f_pred))
            loss_colloc_raw = tf.reduce_mean(tf.square(f_pred))

        T_y_ul = u_derv_y_nd(x_ul, y_ul, t_ul)
        T_x_lr = u_derv_x_nd(x_lr, y_lr, t_lr)
        mse_ulb_sa = bc_w_c * tf.reduce_mean(tf.square(w_ul * T_y_ul))
        mse_ulb = tf.reduce_mean(tf.square(T_y_ul))
        mse_lrb_sa = bc_w_c * tf.reduce_mean(tf.square(w_lr * T_x_lr))
        mse_lrb = tf.reduce_mean(tf.square(T_x_lr))

        total_sa = mse_data_sa + mse_0_sa + loss_colloc_sa + mse_ulb_sa + mse_lrb_sa
        return total_sa, mse_data, mse_0, loss_colloc_raw, mse_ulb + mse_lrb

    @tf.function
    def train_grad(
        x_d, y_d, t_d, T_d,
        x_0, y_0, t_0, T_0,
        x_c, y_c, t_c,
        x_ul, y_ul, t_ul,
        x_lr, y_lr, t_lr,
        w_d_val, w_0_val, w_c_val, w_ul_val, w_lr_val,
    ):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([w_d_val, w_0_val, w_c_val, w_ul_val, w_lr_val])
            total, l_d, l_0, l_c, l_b = compute_loss_nd(
                x_d, y_d, t_d, T_d,
                x_0, y_0, t_0, T_0,
                x_c, y_c, t_c,
                x_ul, y_ul, t_ul,
                x_lr, y_lr, t_lr,
                w_d_val, w_0_val, w_c_val, w_ul_val, w_lr_val,
            )
        grads_net = tape.gradient(total, u_model.trainable_variables)
        grads_d = tape.gradient(total, w_d_val)
        grads_0 = tape.gradient(total, w_0_val)
        grads_c = tape.gradient(total, w_c_val)
        grads_ul = tape.gradient(total, w_ul_val)
        grads_lr = tape.gradient(total, w_lr_val)
        del tape
        return total, l_d, l_0, l_c, l_b, grads_net, grads_d, grads_0, grads_c, grads_ul, grads_lr

    # 4. 初始化点级自适应权重（归一化空间下均从 O(1) 起步）
    weights_data = [
        tf.Variable(tf.ones([batch_size, 1], dtype=tf_dtype))
        for _ in range(num_batches)
    ]
    weights_colloc = [
        tf.Variable(tf.ones([batch_size, 1], dtype=tf_dtype))
        for _ in range(num_batches)
    ]
    weights_0 = tf.Variable(tf.ones([n_init_actual, 1], dtype=tf_dtype))
    weights_ulb = tf.Variable(tf.ones([N_boundary, 1], dtype=tf_dtype))
    weights_lrb = tf.Variable(tf.ones([N_boundary, 1], dtype=tf_dtype))

    tf_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr_net, beta_1=0.9)
    tf_optimizer_weights = tf.keras.optimizers.legacy.Adam(
        learning_rate=args.lr_weights, beta_1=0.9
    )

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=tf_optimizer, net=u_model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"[Ckpt] Restored from {manager.latest_checkpoint}")
    else:
        print("[Ckpt] Initializing from scratch.")

    history = {
        "epoch": [],
        "mse_data_nd": [],
        "mse_0_nd": [],
        "loss_colloc_nd": [],
        "mse_b_nd": [],
        "eval_epoch": [],
        "eval_rmse": [],
        "eval_rel_l2": [],
    }

    # 5. SA-PINN Min-Max 训练循环
    print("Starting Non-Dimensionalized SA-PINN training...")
    for epoch in range(tf_iter):
        epoch_start = time.time()
        progbar = tf.keras.utils.Progbar(num_batches)

        # 每个 epoch 随机打乱 batch 访问顺序，但每个 batch 内部点与自身权重一一绑定
        batch_order = rng.permutation(num_batches)
        l_d_sum = l_0_sum = l_c_sum = l_b_sum = total_sum = 0.0

        for step_i, b_idx in enumerate(batch_order):
            b_idx = int(b_idx)
            X_d_b = data_batches[b_idx]
            X_c_b = colloc_batches[b_idx]
            (
                loss_val, l_d, l_0, l_c, l_b,
                grads_net, g_d, g_0, g_c, g_ul, g_lr,
            ) = train_grad(
                X_d_b[:, 1:2], X_d_b[:, 2:3], X_d_b[:, 0:1], X_d_b[:, 3:4],
                x_0_nd, y_0_nd, t_0_nd, T_0_nd,
                X_c_b[:, 1:2], X_c_b[:, 2:3], X_c_b[:, 0:1],
                x_ul_nd, y_ul_nd, t_ul_nd,
                x_lr_nd, y_lr_nd, t_lr_nd,
                weights_data[b_idx].read_value(),
                weights_0.read_value(),
                weights_colloc[b_idx].read_value(),
                weights_ulb.read_value(),
                weights_lrb.read_value(),
            )
            tf_optimizer.apply_gradients(zip(grads_net, u_model.trainable_variables))
            tf_optimizer_weights.apply_gradients(
                zip(
                    [-g_d, -g_0, -g_c, -g_ul, -g_lr],
                    [
                        weights_data[b_idx],
                        weights_0,
                        weights_colloc[b_idx],
                        weights_ulb,
                        weights_lrb,
                    ],
                )
            )
            l_d_sum += float(l_d.numpy())
            l_0_sum += float(l_0.numpy())
            l_c_sum += float(l_c.numpy())
            l_b_sum += float(l_b.numpy())
            total_sum += float(loss_val.numpy())
            progbar.update(step_i + 1)

        ckpt.step.assign_add(1)
        ep = epoch + 1
        elapsed = time.time() - epoch_start
        avg_d = l_d_sum / num_batches
        avg_0 = l_0_sum / num_batches
        avg_c = l_c_sum / num_batches
        avg_b = l_b_sum / num_batches
        avg_tot = total_sum / num_batches

        history["epoch"].append(ep)
        history["mse_data_nd"].append(avg_d)
        history["mse_0_nd"].append(avg_0)
        history["loss_colloc_nd"].append(avg_c)
        history["mse_b_nd"].append(avg_b)

        print(
            f"Epoch {ep:3d}/{tf_iter} | Time: {elapsed:.2f}s | "
            f"data_nd: {avg_d:.4e} | ic_nd: {avg_0:.4e} | "
            f"pde_nd: {avg_c:.4e} | bc_nd: {avg_b:.4e} | Total_SA: {avg_tot:.4e}"
        )
        manager.save()

        wandb.log(
            {
                "epoch": ep,
                "mse_data_nd": avg_d,
                "mse_0_nd": avg_0,
                "loss_colloc_nd": avg_c,
                "mse_b_nd": avg_b,
                "Total_SA": avg_tot,
            }
        )

        if ep == 1 or ep % plot_every == 0 or ep == tf_iter:
            T_pred_nd = predict_nd(x_test_nd, y_test_nd, t_test_nd).numpy().astype(np.float64)
            T_pred_K = T_ref + delta_T * T_pred_nd

            diff = T_pred_K - T_test_true
            mse_K = float(np.mean(diff ** 2))
            rmse_K = math.sqrt(mse_K)
            rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(T_test_true))

            im_true = np.reshape(T_test_true, (101, 101), order="F")
            im_pred = np.reshape(T_pred_K, (101, 101), order="F")
            pos_true = np.unravel_index(int(np.argmax(im_true)), (101, 101))
            pos_pred = np.unravel_index(int(np.argmax(im_pred)), (101, 101))

            history["eval_epoch"].append(ep)
            history["eval_rmse"].append(rmse_K)
            history["eval_rel_l2"].append(rel_l2)

            print(
                f"[Eval @ t=5.0s] Ep {ep:3d} | RMSE: {rmse_K:.2f} K | Rel-L2: {rel_l2 * 100:.2f}% | "
                f"Peak(True/Pred): {im_true.max():.1f} K / {im_pred.max():.1f} K | "
                f"Pos(True/Pred): {pos_true} / {pos_pred}"
            )
            wandb.log({"eval_rmse_K": rmse_K, "eval_rel_l2": rel_l2})

            out_png = os.path.join(out_dir, f"pred_epoch_{ep}.png")
            save_eval_plot(T_test_true, T_pred_K, ep, rmse_K, rel_l2, out_png)
            print(f"[Plot] Saved {out_png}")

    loss_png = os.path.join(out_dir, "loss_curve.png")
    save_loss_curve(history, loss_png)
    print(f"[Plot] Saved {loss_png}")
    wandb.finish()


if __name__ == "__main__":
    main()
