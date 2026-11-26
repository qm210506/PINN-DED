"""环境与 MOOSE Exodus II (.e) 数据集自检脚本。

放置位置：/home/qm/work/PINNs_Adaptive_thermal/check_env.py
运行方式：python check_env.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: F401
import netCDF4
import numpy as np
import scipy
from scipy.stats import qmc
import tensorflow as tf
import wandb


def main():
    parser = argparse.ArgumentParser(description="检查 PINNs_Adaptive_thermal 复现环境与数据文件")
    parser.add_argument(
        "--data-file",
        type=str,
        default="function_heat_source_out_100W_6s.e",
        help="MOOSE Exodus (.e) 文件路径",
    )
    args = parser.parse_args()

    print("=== 1. Python 与核心库版本 ===")
    print(f"Python     : {sys.version.split()[0]}")
    print(f"TensorFlow : {tf.__version__}")
    print(f"NumPy      : {np.__version__}")
    print(f"SciPy      : {scipy.__version__}")
    print(f"Matplotlib : {matplotlib.__version__} (backend={matplotlib.get_backend()})")
    print(f"netCDF4    : {netCDF4.__version__}")
    print(f"wandb      : {wandb.__version__}")

    print("\n=== 2. 计算设备检测 ===")
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"CPUs: {cpus}")
    print(f"GPUs: {gpus}")
    if not gpus:
        print("[提示] 未检测到 GPU，当前将以 CPU 模式运行，建议先用 `python main_repro.py --mode quick` 验证。")

    print("\n=== 3. SciPy LHS 采样测试 ===")
    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=8)
    print(f"LHS sample shape: {sample.shape}, min={sample.min():.4f}, max={sample.max():.4f}")

    print("\n=== 4. MOOSE Exodus II (.e) 数据文件检测 ===")
    if not os.path.exists(args.data_file):
        print(f"[错误] 找不到数据文件: {args.data_file}")
        print("请确认已在 /home/qm/work/PINNs_Adaptive_thermal 目录下检出 function_heat_source_out_100W_6s.e")
        sys.exit(1)

    with netCDF4.Dataset(args.data_file) as nc:
        if "vals_nod_var1" not in nc.variables:
            print(f"[错误] 文件中缺少 vals_nod_var1 变量，可用变量: {list(nc.variables.keys())}")
            sys.exit(1)
        temps = np.ma.getdata(nc.variables["vals_nod_var1"][:]).astype(np.float64)

    print(f"File path  : {os.path.abspath(args.data_file)}")
    print(f"Raw shape  : {temps.shape} (预期: (121, 61206))")
    grid = temps.reshape((121, 6, 101, 101))
    top_slice = grid[:, 5, :, :]
    print(f"Top slice  : {top_slice.shape} (t, x, y)")
    print(f"Temp range : min={top_slice.min():.2f} K, max={top_slice.max():.2f} K")
    print("\n[OK] 环境与数据自检全部通过！")


if __name__ == "__main__":
    main()
