import os

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
os.chdir("/home/cz06540/concept-prune/wanda")
from utils import get_prompts, Config, get_sd_model




def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default="golf ball")
    parser.add_argument('--base', type=str, default="a room")
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=0.02)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default=None)
    return parser.parse_args()




args = input_args()

SPECIFIC_FILES = []
# INPUT_DIR = Path(f"/scratch/cz06540/concept-prune-csv/{args.target}/pruned_csv").resolve()
INPUT_DIR = Path(f"/scratch/cz06540/defense_rebuttal_csv/{args.target}/pruned_csv").resolve()


print(args.target)
print(INPUT_DIR)
OUTPUT_DIR = INPUT_DIR.parent / "output" / "GPU_500it_1e5cv_rankNone_ft64"
# OUTPUT_DIR = INPUT_DIR.parent / "output" / "GPU_500it_1e5cv_rankNone_ft64"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOFTIMPUTE_KWARGS = dict(
    max_iters=500,
    convergence_threshold=1e-5,
    init_fill_method="mean",
    n_power_iterations=2,
    verbose=True,
)


def read_csv_as_float_matrix(csv_path: Path) -> np.ndarray:
    df_raw = pd.read_csv(csv_path, header=None, dtype=str)
    df = df_raw.apply(pd.to_numeric, errors='coerce')
    return df.values.astype(float)

def softimpute_complete(arr: np.ndarray) -> np.ndarray:
    max_iters = SOFTIMPUTE_KWARGS.get("max_iters", 100)
    tol = SOFTIMPUTE_KWARGS.get("convergence_threshold", 1e-4)
    init_fill_method = SOFTIMPUTE_KWARGS.get("init_fill_method", "mean")
    max_rank = SOFTIMPUTE_KWARGS.get("max_rank", None)
    verbose = SOFTIMPUTE_KWARGS.get("verbose", False)

    valid_mask = ~np.isnan(arr)
    zeros_mask = (arr == 0.0) & valid_mask

    X_missing = arr.copy().astype(float)
    X_missing[zeros_mask] = np.nan
    if np.isnan(X_missing).all():
        return arr

    
    print("尝试使用 GPU 版本的 SoftImpute ...")
    
    import cupy as cp

    X = cp.asarray(X_missing, dtype=cp.float64)

    obs_mask = cp.asarray(~cp.isnan(X))
    X_filled = X.copy()
    if init_fill_method == "mean":
        col_means = cp.nanmean(X_filled, axis=0)
        inds = cp.where(cp.isnan(X_filled))
        X_filled[inds] = cp.take(col_means, inds[1])
    else:
        X_filled = cp.nan_to_num(X_filled, nan=0.0)

    def svt_shrink(M, rank_cap=None, tau=None):
        U, s, VT = cp.linalg.svd(M, full_matrices=False)
        if rank_cap is not None and rank_cap > 0 and rank_cap < s.shape[0]:
            U = U[:, :rank_cap]
            s = s[:rank_cap]
            VT = VT[:rank_cap, :] 
        if tau is None:
            if s.shape[0] > 0:
                tau = s[-1]
            else:
                tau = 0.0
        s_shrunk = cp.maximum(s - tau, 0.0)
        r = int(cp.count_nonzero(s_shrunk).get())
        if r == 0:
            return cp.zeros_like(M), 0
        U_r = U[:, :r]
        VT_r = VT[:r, :]
        S_r = cp.diag(s_shrunk[:r])
        return U_r @ S_r @ VT_r, r

    X_obs = X.copy()

    prev = None
    for it in range(1, max_iters + 1):
        X_lowrank, eff_rank = svt_shrink(X_filled, rank_cap=max_rank, tau=None)

        X_new = X_lowrank
        X_new = cp.where(obs_mask, X_obs, X_new)

        if prev is not None:
            denom = cp.linalg.norm(prev)
            diff = cp.linalg.norm(X_new - prev)
            rel_change = (diff / (denom + 1e-12)).item()
        else:
            rel_change = cp.inf

        mae = cp.mean(cp.abs(cp.where(obs_mask, X_new - X_obs, 0.0)) ).item()

        if verbose:
            print(f"[SoftImpute-GPU] Iter {it}: observed MAE={mae:.6f} rank={eff_rank} rel_change={rel_change:.6e}")

        if rel_change < tol:
            X_filled = X_new
            break

        prev = X_filled
        X_filled = X_new

    X_completed = cp.asnumpy(X_filled)

    observed_mask_np = ~np.isnan(X_missing)
    X_completed[observed_mask_np] = arr[observed_mask_np]

    print(f"GPU SoftImpute 完成，共 {it} 轮，最终观测位 MAE={mae:.6f}")
    return X_completed

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"未找到输入目录：{INPUT_DIR}")

    files = [INPUT_DIR / f for f in SPECIFIC_FILES] if SPECIFIC_FILES else sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        print("pruned_csv 目录下没有找到任何 .csv 文件。")
        return

    print(f"共找到 {len(files)} 个 CSV，将依次处理。输出目录：{OUTPUT_DIR}")

    for csv_path in files:
        print(f"Processing {csv_path.name} ...")
        try:
            arr = read_csv_as_float_matrix(csv_path)
            soft_path = OUTPUT_DIR / f"SoftImpute_{csv_path.name}"

            if soft_path.exists():
                print(f"[跳过] {csv_path.name} 已有结果")
                continue

            X_soft = softimpute_complete(arr)
            pd.DataFrame(X_soft).to_csv(OUTPUT_DIR / f"SoftImpute_{csv_path.name}", header=False, index=False)


        except Exception as e:
            print(f"[失败] {csv_path.name}: {e}")

if __name__ == "__main__":
    main()
