"""SA-PINN 二维瞬态移动热源传热模型复现脚本（TensorFlow 2.15 + WSL2 Ubuntu 22.04 兼容版）。

放置位置：/home/qm/work/PINNs_Adaptive_thermal/main_repro.py

相较于原仓库 main.py 的核心修复与改进：
1. 使用 matplotlib Agg 后端，适配无图形界面的 WSL2 环境。
2. 修复 TensorFlow 2.15 移除 Adam(lr=...) 的问题，改为 learning_rate=...。
3. 使用 tf.keras.optimizers.legacy.Adam 并通过 read_value() + tape.watch() 传参，
   同时解决 Keras 2.11+ 动态权重变量报错 (Unknown variable) 与 @tf.function 反复重编译问题。
4. 用 scipy.stats.qmc.LatinHypercube 替代已停更的 pyDOE（避免 setuptools>=58 安装失败）。
5. 默认关闭 wandb 在线登录阻塞（可通过 --wandb-mode 切换）。
6. 仅加载一次 .e 数据文件并复用插值器（原代码重复加载了 5 次）。
7. 独立 --ckpt-dir 避免误读原仓库旧版 TF 2.5 checkpoints，并新增 --mode quick/full 方便 CPU 冒烟验证。
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

# 统一默认双精度浮点，减少高阶导数与热源大系数数值误差
tf.keras.backend.set_floatx("float64")


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


def build_neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],), dtype="float64"))
    for width in layer_sizes[1:-1]:
        model.add(
            layers.Dense(
                width,
                activation=tf.nn.tanh,
                kernel_initializer="glorot_normal",
                dtype="float64",
            )
        )
    model.add(
        layers.Dense(
            layer_sizes[-1],
            activation=None,
            kernel_initializer="glorot_normal",
            dtype="float64",
        )
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="SA-PINN Thermal Reproduction (TF 2.15)")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: CPU 快速验证模式; full: 对齐原仓库 main.py 全量规模参数",
    )
    parser.add_argument("--epochs", type=int, default=None, help="自定义训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--data-file",
        type=str,
        default="function_heat_source_out_100W_6s.e",
        help="MOOSE Exodus II (.e) 文件路径",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint 保存目录（默认按 mode 自动区分 ./tf_ckpts_quick 或 ./tf_ckpts_full）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="预测图输出目录（默认按 mode 自动区分 ./outputs_quick 或 ./outputs_full）",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="WandB 日志模式（默认 disabled 避免登录阻塞）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.mode == "quick":
        layer_sizes = [3, 64, 64, 64, 1]
        batch_size = 512
        N_colloc = 4096
        N_data = 4096
        N_initial = 500
        N_boundary = 500
        tf_iter = 5 if args.epochs is None else args.epochs
        plot_every = 1
        ckpt_dir = args.ckpt_dir or "./tf_ckpts_quick"
        out_dir = args.out_dir or "./outputs_quick"
    else:
        layer_sizes = [3, 128, 128, 128, 128, 128, 1]
        batch_size = 2048
        N_colloc = 256000
        N_data = 256000
        N_initial = 5000
        N_boundary = 5000
        tf_iter = 10 if args.epochs is None else args.epochs
        plot_every = 10
        ckpt_dir = args.ckpt_dir or "./tf_ckpts_full"
        out_dir = args.out_dir or "./outputs_full"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    num_batches = N_colloc // batch_size
    print(
        f"[Config] mode={args.mode} | layers={layer_sizes} | epochs={tf_iter} | "
        f"N_data={N_data} | N_colloc={N_colloc} | batch_size={batch_size} ({num_batches} batches/epoch)"
    )
    print(f"[Paths ] data={args.data_file} | ckpt={ckpt_dir} | out={out_dir}")
    print(f"[Device] {tf.config.list_physical_devices()}")

    wandb.init(project="sa-pinns", mode=args.wandb_mode)

    # 1. 构建网络与物理控制方程残差算子
    u_model = build_neural_net(layer_sizes)

    rho = tf.constant(8e-3, dtype=tf.float64)
    cp = tf.constant(500e4, dtype=tf.float64)
    k = tf.constant(16.3e2, dtype=tf.float64)
    r = tf.constant(1e-2, dtype=tf.float64)
    coeff_1 = tf.constant(6.0, dtype=tf.float64)
    coeff_2 = tf.constant(math.sqrt(3.0 / (math.pi ** 3)), dtype=tf.float64)
    coeff_3 = tf.constant(7e11, dtype=tf.float64)
    heat_source_coeff = coeff_1 * coeff_2 * coeff_3

    @tf.function
    def f_model(x, y, t):
        T = u_model(tf.concat([x, y, t], 1))
        T_x = tf.gradients(T, x)[0]
        T_y = tf.gradients(T, y)[0]
        T_xx = tf.gradients(T_x, x)[0]
        T_yy = tf.gradients(T_y, y)[0]
        T_t = tf.gradients(T, t)[0]

        fx = tf.constant(1.8e-1, dtype=tf.float64) * t
        c1 = (x - fx) / r
        c2 = (y - tf.constant(0.5, dtype=tf.float64)) / r

        return (
            rho * cp * T_t
            - k * (T_xx + T_yy)
            - heat_source_coeff
            * tf.math.exp(tf.constant(-3.0, dtype=tf.float64) * (c1 ** 2 + c2 ** 2))
        )

    @tf.function
    def u_derv_ulb_model(x, y, t):
        T = u_model(tf.concat([x, y, t], 1))
        return tf.gradients(T, y)[0]

    @tf.function
    def u_derv_lrb_model(x, y, t):
        T = u_model(tf.concat([x, y, t], 1))
        return tf.gradients(T, x)[0]

    def compute_loss(
        x_d, y_d, t_d, T_d,
        x_0, y_0, t_0, T_0,
        x_c, y_c, t_c,
        x_ul, y_ul, t_ul,
        x_lr, y_lr, t_lr,
        w_d, w_0, w_c, w_ul, w_lr,
    ):
        T_d_pred = u_model(tf.concat([x_d, y_d, t_d], 1))
        mse_data_sa = tf.reduce_mean(tf.square(w_d * (T_d - T_d_pred)))
        mse_data = tf.reduce_mean(tf.square(T_d - T_d_pred))

        T_0_pred = u_model(tf.concat([x_0, y_0, t_0], 1))
        mse_0_sa = tf.reduce_mean(tf.square(w_0 * (T_0 - T_0_pred)))
        mse_0 = tf.reduce_mean(tf.square(T_0 - T_0_pred))

        f_pred = f_model(x_c, y_c, t_c)
        mse_colloc_sa = tf.reduce_mean(tf.square(w_c * f_pred))
        mse_colloc = tf.reduce_mean(tf.square(f_pred))

        T_y_ul = u_derv_ulb_model(x_ul, y_ul, t_ul)
        T_x_lr = u_derv_lrb_model(x_lr, y_lr, t_lr)
        mse_ulb_sa = tf.reduce_mean(tf.square(w_ul * T_y_ul))
        mse_ulb = tf.reduce_mean(tf.square(T_y_ul))
        mse_lrb_sa = tf.reduce_mean(tf.square(w_lr * T_x_lr))
        mse_lrb = tf.reduce_mean(tf.square(T_x_lr))

        total_sa = mse_data_sa + mse_0_sa + mse_colloc_sa + mse_ulb_sa + mse_lrb_sa
        return total_sa, mse_data, mse_0, mse_colloc, mse_ulb, mse_lrb

    @tf.function
    def train_grad(
        x_d, y_d, t_d, T_d,
        x_0, y_0, t_0, T_0,
        x_c, y_c, t_c,
        x_ul, y_ul, t_ul,
        x_lr, y_lr, t_lr,
        w_d_val, w_0_val, w_c_val, w_ul_val, w_lr_val,
    ):
        # 通过纯 Tensor 值 + tape.watch() 传参，确保所有 batch 共用同一份编译图（零 retracing）
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([w_d_val, w_0_val, w_c_val, w_ul_val, w_lr_val])
            total, l_d, l_0, l_c, l_ul, l_lr = compute_loss(
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
        return (
            total, l_d, l_0, l_c, l_ul, l_lr,
            grads_net, grads_d, grads_0, grads_c, grads_ul, grads_lr,
        )

    # 2. 单次加载 MOOSE Exodus 文件并构建 5 类数据集
    print(f"[Data] Loading {args.data_file} ...")
    t_load = time.time()
    temp_data = DataPrep(xmax=1.0, filename=args.data_file)

    X_data_coords = temp_data.get_coords(
        xmin=0.01, xmax=0.99, ymin=0.01, ymax=0.99, tmin=0.05, num_x=99, num_y=99, num_t=96
    )
    X_data = temp_data.get_temps(X_data_coords)
    X_dataset_batch = (
        tf.data.Dataset.from_tensor_slices(X_data)
        .shuffle(buffer_size=10000, seed=args.seed)
        .take(N_data)
        .batch(batch_size, drop_remainder=True)
    )

    X_0_all = temp_data.get_temps(
        temp_data.get_coords(0.01, 0.99, 0.01, 0.99, 0.0, 99, 99, 1)
    )
    X_0 = X_0_all[np.random.randint(X_0_all.shape[0], size=N_initial), :]

    # 使用 SciPy 内置拉丁超立方采样替代已停更的 pyDOE
    lb = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ub = np.array([4.85, 1.0, 1.0], dtype=np.float64)
    lhs_engine = qmc.LatinHypercube(d=3, seed=args.seed)
    X_colloc_np = qmc.scale(lhs_engine.random(n=N_colloc), lb, ub)
    X_colloc_batch = (
        tf.data.Dataset.from_tensor_slices(X_colloc_np)
        .shuffle(buffer_size=10000, seed=args.seed)
        .take(N_colloc)
        .batch(batch_size, drop_remainder=True)
    )

    X_ul_all = temp_data.get_temps(
        temp_data.get_coords(0.01, 0.99, 0.0, 1.0, 0.05, 99, 2, 96)
    )
    X_ul = X_ul_all[np.random.randint(X_ul_all.shape[0], size=N_boundary), :]

    X_lr_all = temp_data.get_temps(
        temp_data.get_coords(0.0, 1.0, 0.01, 0.99, 0.05, 2, 99, 96)
    )
    X_lr = X_lr_all[np.random.randint(X_lr_all.shape[0], size=N_boundary), :]

    X_test = temp_data.get_temps(
        temp_data.get_coords(0.0, 1.0, 0.0, 1.0, 5.0, 101, 101, 1),
        temp_cutoff=50000.0,
    )
    print(f"[Data] Ready in {time.time() - t_load:.2f}s.")

    x_0 = tf.constant(X_0[:, 1:2], dtype=tf.float64)
    y_0 = tf.constant(X_0[:, 2:3], dtype=tf.float64)
    t_0 = tf.constant(X_0[:, 0:1], dtype=tf.float64)
    T_0 = tf.constant(X_0[:, 3:4], dtype=tf.float64)

    x_ulb = tf.constant(X_ul[:, 1:2], dtype=tf.float64)
    y_ulb = tf.constant(X_ul[:, 2:3], dtype=tf.float64)
    t_ulb = tf.constant(X_ul[:, 0:1], dtype=tf.float64)

    x_lrb = tf.constant(X_lr[:, 1:2], dtype=tf.float64)
    y_lrb = tf.constant(X_lr[:, 2:3], dtype=tf.float64)
    t_lrb = tf.constant(X_lr[:, 0:1], dtype=tf.float64)

    x_test = tf.constant(X_test[:, 1:2], dtype=tf.float64)
    y_test = tf.constant(X_test[:, 2:3], dtype=tf.float64)
    t_test = tf.constant(X_test[:, 0:1], dtype=tf.float64)
    T_test = X_test[:, 3:4]

    X_data_colloc_batch = tf.data.Dataset.zip((X_dataset_batch, X_colloc_batch))

    # 3. 初始化点级自适应权重与兼容版 Adam 优化器
    weights_data = [
        tf.Variable(tf.random.uniform([batch_size, 1], dtype="float64"))
        for _ in range(num_batches)
    ]
    weights_colloc = [
        tf.Variable(tf.random.uniform([batch_size, 1], dtype="float64"))
        for _ in range(num_batches)
    ]
    weights_0 = tf.Variable(100.0 * tf.random.uniform([N_initial, 1], dtype="float64"))
    weights_ulb = tf.Variable(tf.random.uniform([N_boundary, 1], dtype="float64"))
    weights_lrb = tf.Variable(tf.random.uniform([N_boundary, 1], dtype="float64"))

    tf_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.99)
    tf_optimizer_weights = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.99)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=tf_optimizer, net=u_model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"[Ckpt] Restored from {manager.latest_checkpoint}")
    else:
        print("[Ckpt] Initializing from scratch.")

    # 4. SA-PINN Min-Max 交替训练循环
    print("Starting SA-PINN training...")
    for epoch in range(tf_iter):
        epoch_start = time.time()
        progbar = tf.keras.utils.Progbar(num_batches)
        for batch_idx, (X_d_b, X_c_b) in enumerate(X_data_colloc_batch):
            (
                loss_val, l_d, l_0, l_c, l_ul, l_lr,
                grads_net, g_d, g_0, g_c, g_ul, g_lr,
            ) = train_grad(
                X_d_b[:, 1:2], X_d_b[:, 2:3], X_d_b[:, 0:1], X_d_b[:, 3:4],
                x_0, y_0, t_0, T_0,
                X_c_b[:, 1:2], X_c_b[:, 2:3], X_c_b[:, 0:1],
                x_ulb, y_ulb, t_ulb,
                x_lrb, y_lrb, t_lrb,
                weights_data[batch_idx].read_value(),
                weights_0.read_value(),
                weights_colloc[batch_idx].read_value(),
                weights_ulb.read_value(),
                weights_lrb.read_value(),
            )
            tf_optimizer.apply_gradients(zip(grads_net, u_model.trainable_variables))
            tf_optimizer_weights.apply_gradients(
                zip(
                    [-g_d, -g_0, -g_c, -g_ul, -g_lr],
                    [
                        weights_data[batch_idx],
                        weights_0,
                        weights_colloc[batch_idx],
                        weights_ulb,
                        weights_lrb,
                    ],
                )
            )
            progbar.update(batch_idx + 1)

        ckpt.step.assign_add(1)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{tf_iter} | Time: {elapsed:.2f}s | "
            f"mse_data: {l_d.numpy():.4e} | mse_0: {l_0.numpy():.4e} | "
            f"mse_colloc: {l_c.numpy():.4e} | mse_b: {(l_ul + l_lr).numpy():.4e} | "
            f"Total_SA: {loss_val.numpy():.4e}"
        )
        save_path = manager.save()
        print(f"[Ckpt] Saved step {int(ckpt.step)}: {save_path}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss_data": float(l_d.numpy()),
                "loss_0": float(l_0.numpy()),
                "loss_colloc": float(l_c.numpy()),
                "loss_b": float((l_ul + l_lr).numpy()),
                "Total_loss": float(loss_val.numpy()),
            }
        )

        if (epoch + 1) % plot_every == 0 or (epoch + 1) == tf_iter:
            T_test_pred = u_model(tf.concat([x_test, y_test, t_test], 1)).numpy()
            test_mse = float(np.mean(np.square(T_test - T_test_pred)))
            print(f"[Eval @ t=5.0s] Epoch {epoch + 1} Test MSE: {test_mse:.4f}")

            im1 = np.reshape(T_test, (101, 101), order="F")
            im2 = np.reshape(T_test_pred, (101, 101), order="F")
            im3 = np.abs(im2 - im1)

            fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3.5), ncols=3)
            pos = ax1.imshow(im1, cmap="jet", interpolation="none")
            fig.colorbar(pos, ax=ax1, extend="both", label="Temperature (K)")
            ax1.set_title("MOOSE Reference (t=5.0s)")

            neg = ax2.imshow(im2, cmap="jet", interpolation="none")
            fig.colorbar(neg, ax=ax2, extend="both", label="Temperature (K)")
            ax2.set_title(f"SA-PINN Prediction (Epoch {epoch + 1})")

            err = ax3.imshow(im3, cmap="jet", interpolation="none")
            fig.colorbar(err, ax=ax3, extend="both", label="Abs Error (K)")
            ax3.set_title("Absolute Error Map")

            fig.tight_layout()
            out_png = os.path.join(out_dir, f"pred_epoch_{epoch + 1}.png")
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[Plot] Saved {out_png}")

    wandb.finish()


if __name__ == "__main__":
    main()
