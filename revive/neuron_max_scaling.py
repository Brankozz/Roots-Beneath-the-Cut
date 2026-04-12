import copy
import os
import json
import scipy
import pickle
import torch
import sys
import tqdm
import numpy as np
from argparse import ArgumentParser
import random
from PIL import Image, ImageFilter, ImageDraw
import re
from diffusers.pipelines.stable_diffusion import safety_checker
import matplotlib.pyplot as plt
from sentry_sdk.profiler import MAX_PROFILE_DURATION_NS

from utils import get_prompts, Config, get_sd_model
os.chdir("/home/cz06540/concept-prune/wanda")
import seaborn as sns
import bitsandbytes as bnb
from neuron_receivers import Wanda
from transformers.models.clip.modeling_clip import CLIPMLP
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import math
import pandas as pd
import glob

SOFTIMPUTE_KWARGS = dict(
    max_iters=100,
    convergence_threshold=1e-4,
    init_fill_method="mean",
    max_rank=50,
    n_power_iterations=2,
    verbose=True,
)

FFN2_PATTERN = re.compile(r"(^|\.)(ff\.net\.2)$")

def _is_ffn_fc2(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and FFN2_PATTERN.search(name) is not None


def softimpute_complete(arr: np.ndarray) -> np.ndarray:
    from fancyimpute import SoftImpute
    valid_mask = ~np.isnan(arr)
    zeros_mask = (arr == 0.0) & valid_mask

    X_missing = arr.copy().astype(float)
    X_missing[zeros_mask] = np.nan

    if np.isnan(X_missing).all():
        return arr

    # imputer = SoftImpute(**SOFTIMPUTE_KWARGS)
    imputer = SoftImpute(**{**SOFTIMPUTE_KWARGS, "verbose": True})

    X_completed = imputer.fit_transform(X_missing)

    observed_mask = ~np.isnan(X_missing)
    X_completed[observed_mask] = arr[observed_mask]
    return X_completed

def _softimpute_fill_all_zeros(linear: nn.Linear) -> bool:
    """把 linear.weight 里所有 ==0 的元素当缺失，用 SoftImpute 补全。"""
    W = linear.weight.detach()
    device, dtype = W.device, W.dtype

    W_np = W.float().cpu().numpy()
    zero_mask = (W_np == 0.0)
    if not zero_mask.any():
        return False

    X_missing = W_np.copy()
    X_missing[zero_mask] = np.nan
    if np.isnan(X_missing).all():
        return False

    X_completed = softimpute_complete(X_missing)
    W_new = torch.from_numpy(X_completed)

    with torch.no_grad():
        linear.weight.copy_(W_new.to(device=device, dtype=dtype))
    return True




def softimpute_unet_ffn2(unet: nn.Module) -> int:
    """
    遍历整个 UNet，把所有 FFN 第二个全连接层 (ff.net.2.weight)
    的 0 元素用 SoftImpute 补全。

    返回成功补全的层数。
    """
    filled_count = 0
    for name, module in unet.named_modules():
        if _is_ffn_fc2(name, module):
            ok = _softimpute_fill_all_zeros(module)
            if ok:
                filled_count += 1
                print(f"[SoftImpute] filled zeros in layer: {name}")
    return filled_count















def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default="english springer")
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=0.02)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default=None)
    parser.add_argument('--top_ratio', type=str, default=None)
    parser.add_argument('--magnitude_process', type=str, default="Max")
    parser.add_argument('--csv_folder', type=str, default='/scratch/cz06540/concept-prune-csv/golf ball/top_ratio_output')
    return parser.parse_args()


def collect_weights(module):
    weights = []
    for name, param in module.named_parameters():
        if param.requires_grad and param.dim() > 0:
            weights.append(param.detach().cpu().flatten())
    return torch.cat(weights)


@torch.no_grad()
def restore_pruned_weights_rowwise_dynamic_align_sign(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    Mag_seed = 2021,
    Sign_seed = 234,
    align_prob: float = 1.0,
    global_sample = True,
    Max_processing = True,
    Ave_processing = False,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if Mag_seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(Mag_seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "used_fallback_orig_row": 0,
    }
    per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]                     # will be modified in-place on its device
        wb = pb.detach().cpu()

        # _softimpute_fill_all_zeros(wb)

        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        # layer-level fallbacks (μ/σ) & layer max|.| for the hard constraint
        # keep_layer = wb.abs() > eps_pruned
        # layer_keep_cnt = int(keep_layer.sum().item())
        # layer_mu = layer_sigma = None
        # if layer_keep_cnt >= layer_min_samples:
        #     vals = wb[keep_layer].float()
        #     layer_mu    = vals.mean().item()
        #     layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        # layer_max = vals.abs().max().item() if layer_keep_cnt > 0 else None

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]

            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            row_vals = prun_row[keep_row].float()
            mu = row_vals.mean().item()
            sigma = float(row_vals.std(unbiased=False).item() or 1e-8)

            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu  # N(mu, sigma^2)
            mags = samples.abs()
            if Max_processing:
                row_max = float(prun_row[keep_row].abs().max().item())
                margin = max(gt_abs_margin, row_max * gt_rel_margin)
                hard_thresh = row_max + margin
                mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            if Ave_processing:
                row_mean = float(prun_row[keep_row].abs().mean().item())
                mags = row_mean

            sign_orig = torch.sign(orig_row[pruned_mask_row]).float()
            sign_samp = torch.sign(samples).float()



            sign_use  = torch.where(sign_orig == 0, sign_samp, sign_orig)

            new_vals = mags * sign_use


            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}




@torch.no_grad()
def restore_pruned_weights_rowwise_align_magnitudes(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    Mag_seed = 2021,
    Sign_seed = 234,
    align_prob: float = 1.0,
    global_sample = True,
    Max_processing = True,
    Ave_processing = False,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if Mag_seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(Mag_seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "used_fallback_orig_row": 0,
    }
    per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]                     # will be modified in-place on its device
        wb = pb.detach().cpu()

        # _softimpute_fill_all_zeros(wb)

        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        # layer-level fallbacks (μ/σ) & layer max|.| for the hard constraint
        # keep_layer = wb.abs() > eps_pruned
        # layer_keep_cnt = int(keep_layer.sum().item())
        # layer_mu = layer_sigma = None
        # if layer_keep_cnt >= layer_min_samples:
        #     vals = wb[keep_layer].float()
        #     layer_mu    = vals.mean().item()
        #     layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        # layer_max = vals.abs().max().item() if layer_keep_cnt > 0 else None

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]

            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)          # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned

            row_vals = prun_row[keep_row].float()
            mu = row_vals.mean().item()
            sigma = float(row_vals.std(unbiased=False).item() or 1e-8)

            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu

            sign_samp = torch.sign(samples).float()

            mags = orig_row[pruned_mask_row].abs()
            sign_orig = torch.sign(orig_row[pruned_mask_row].float())


            sign_use  = torch.where(sign_orig == 0, sign_orig, sign_samp)

            new_vals = mags * sign_use



            # write back to device
            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}












@torch.no_grad()
def restore_pruned_weights_rowwise_align_sign_with_activation_sort(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    Mag_seed = 123,
    align_prob: float = 1.0,
    global_sample = True,
    Max_processing = True,
    Ave_processing = True,
    bottom_keep = True,
    opposite_set_zero = True,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if Mag_seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(Mag_seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "used_fallback_orig_row": 0,
    }
    per_param: Dict[str, Any] = {}

    norm_path = f'./results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/{args.target}'
    act_norms_base = torch.load(os.path.join(norm_path, 'base_norms.pt'))
    act_norms_target = torch.load(os.path.join(norm_path, 'target_norms.pt'))

    for idx, name in enumerate(target_names):
        wa = a_params[name].detach().cpu()
        pb = b_params[name]
        wb = pb.detach().cpu()



        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        accumlated_activation_base = sum(act_norms_base[t][idx] for t in range(10))
        accumlated_activation_target = sum(act_norms_target[t][idx] for t in range(10))

        accumlated_ave_diff_activation = (accumlated_activation_target - accumlated_activation_base) / 10

        salience_score_diff = wa.abs() * accumlated_ave_diff_activation


        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]
            salience_score_diff_row = salience_score_diff[r]



            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)          # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            # ---- choose (mu, sigma): row -> layer -> global -> orig_row ----
            # fallback = None
            # if keep_cnt_row >= row_min_samples:

            row_vals = prun_row[keep_row].float()
            mu = row_vals.mean().item()
            sigma = float(row_vals.std(unbiased=False).item() or 1e-8)

            # row_max = float(prun_row[keep_row].abs().max().item())
            # margin = max(gt_abs_margin, row_max * gt_rel_margin)
            # hard_thresh = row_max + margin  # ensure strict '>' in float32

            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu  # N(mu, sigma^2)
            mags = samples.abs()
            if Max_processing:
                row_max = float(prun_row[keep_row].abs().max().item())
                margin = max(gt_abs_margin, row_max * gt_rel_margin)
                hard_thresh = row_max + margin
                mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))
            if Ave_processing:
                row_mean = float(prun_row[keep_row].abs().mean().item())
                mags = row_mean

            # random.seed(Sign_seed)
            # torch.manual_seed(Sign_seed)
            # np.random.seed(Sign_seed)

            sign_orig = torch.sign(orig_row[pruned_mask_row]).float()# CPU {-1,0,+1}
            salience = salience_score_diff_row[pruned_mask_row]
            sorted_idx = torch.argsort(salience, descending=True)
            k = int(align_prob * len(salience))

            if bottom_keep:
                keep_idx = sorted_idx[-k:]
                flip_idx = sorted_idx[:-k]
            else:
                keep_idx = sorted_idx[:k]
                flip_idx = sorted_idx[k:]


            sign_orig[flip_idx] = -sign_orig[flip_idx]

            sign_samp = torch.sign(samples).float()

            sign_use  = torch.where(sign_orig == 0, sign_samp, sign_orig)

            new_vals = mags * sign_use # CPU values after constraint + sign align

            if opposite_set_zero:
                new_vals[flip_idx] = 0

            # write back to device
            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}


@torch.no_grad()
def restore_pruned_weights_with_csv(
        unet_pruned: torch.nn.Module,
        path: str = None,
        eps_orig: float = 0.0,
        eps_pruned: float = 0.0,
        ffn2_only: bool = True,
        row_min_samples: int = 16,
        layer_min_samples: int = 128,
        gt_rel_margin: float = 1e-6,
        gt_abs_margin: float = 1e-12,
        Mag_seed=123,
        global_sample=True,
        Max_processing=False,
        Ave_processing=False,
        CSV_Align=False,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if Mag_seed is not None:
        g = torch.Generator(device="cpu");
        g.manual_seed(Mag_seed)

    # a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())


    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(b_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)


    for idx, name in enumerate(target_names):
        # weight_path = f"../GPU_200it_1e5cv_rankNone/{name}.csv"
        weight_path = os.path.join(path, f"{name}.csv")

        weight_df = pd.read_csv(weight_path, header=None)

        weight_tensor = torch.tensor(weight_df.values, dtype=torch.float32)


        pb = b_params[name]
        wb = pb.detach().cpu()


        out_features, in_features = wb.shape
        device = pb.device
        for r in range(out_features):

            prun_row = wb[r]
            csv_row = weight_tensor[r]



            # mask_orig_nz = orig_row.abs() > eps_orig
            pruned_mask_row = prun_row.abs() <= eps_pruned
            # pruned_mask_row = (mask_orig_nz & mask_prun_z)  # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue



            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            # ---- choose (mu, sigma): row -> layer -> global -> orig_row ----
            # fallback = None
            # if keep_cnt_row >= row_min_samples:

            row_vals = prun_row[keep_row].float()
            mu = row_vals.mean().item()
            sigma = float(row_vals.std(unbiased=False).item() or 1e-8)


            # row_max = float(prun_row[keep_row].abs().max().item())
            # margin = max(gt_abs_margin, row_max * gt_rel_margin)
            # hard_thresh = row_max + margin  # ensure strict '>' in float32
            # ---- sample values for this row's pruned positions ----


            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu  #N(mu, sigma^2)
            mags = samples.abs()


            if Max_processing:
                row_max = float(prun_row[keep_row].abs().max().item())
                margin = max(gt_abs_margin, row_max * gt_rel_margin)
                hard_thresh = row_max + margin
                mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))
            if Ave_processing:
                row_mean = float(prun_row[keep_row].abs().mean().item())
                mags = row_mean

            if CSV_Align:
                mags = csv_row[pruned_mask_row].abs()




            sign_csv = torch.sign(csv_row[pruned_mask_row]).float()

            # sign_samp = torch.sign(samples).float()

            sign_use = torch.where(sign_csv == 0, torch.tensor(0.0, device=sign_csv.device), sign_csv)

            new_vals = mags * sign_use


            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

        # filled_this_param += pruned_cnt_row

        # affected_rows += 1

        # if filled_this_param > 0:

        #     summary["affected_params"] += 1
        #     summary["affected_rows"] += affected_rows
        #     summary["filled_positions"] += filled_this_param
        #     per_param[name] = {"rows_filled": affected_rows,
        #                        "filled_positions": filled_this_param,
        #                        "shape": tuple(wb.shape)}
        # else:
        #     per_param[name] = {"skipped": True, "Reason": "no_pruned_positions"}

    # return {"summary": summary, "per_param": per_param}

        # random.seed(Sign_seed)
        # torch.manual_seed(Sign_seed)
        # np.random.seed(Sign_seed)

        # pb = b_params[name]
        # wb = pb.detach().cpu()

        # if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
        #     per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
        #     continue

        # device = pb.device
        # out_features, in_features = wb.shape
        # filled_this_param = 0
        # affected_rows = 0
        #
        # accumlated_activation_base = sum(act_norms_base[t][idx] for t in range(10))
        # accumlated_activation_target = sum(act_norms_target[t][idx] for t in range(10))
        #
        # accumlated_ave_diff_activation = (accumlated_activation_target - accumlated_activation_base) / 10
        #
        # salience_score_diff = wa.abs() * accumlated_ave_diff_activation



        # sign_orig = torch.sign(orig_row[pruned_mask_row]).float()  # CPU {-1,0,+1}
        # salience = salience_score_diff_row[pruned_mask_row]
        # sorted_idx = torch.argsort(salience, descending=True)
        # k = int(align_prob * len(salience))
        #
        # if bottom_keep:
        #     keep_idx = sorted_idx[-k:]
        #     flip_idx = sorted_idx[:-k]
        # else:
        #     keep_idx = sorted_idx[:k]
        #     flip_idx = sorted_idx[k:]
        #
        # sign_orig[flip_idx] = -sign_orig[flip_idx]
        #





























@torch.no_grad()
def restore_pruned_weights_rowwise_strict_align_sign(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    seed: Optional[int] = 123,
    align_prob: float = 1.0,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    # global_pool = []
    # for n in target_names:
    #     wb = b_params[n].detach().cpu()
    #     if wb.ndim != 2: continue
    #     keep = wb.abs() > eps_pruned
    #     if keep.any(): global_pool.append(wb[keep].float())
    # global_mean = global_std = global_max = None
    # if global_pool:
    #     pooled = torch.cat(global_pool)
    #     global_mean = pooled.mean().item()
    #     global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
    #     global_max  = pooled.abs().max().item()

    # summary = {
    #     "affected_params": 0,
    #     "affected_rows": 0,
    #     "filled_positions": 0,
    #     "used_fallback_layer": 0,
    #     "used_fallback_global": 0,
    #     "used_fallback_orig_row": 0,
    # }
    # per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]                     # will be modified in-place on its device
        wb = pb.detach().cpu()

        # if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
        #     per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
        #     continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        # layer-level fallbacks (μ/σ) & layer max|.| for the hard constraint
        keep_layer = wb.abs() > eps_pruned
        layer_keep_cnt = int(keep_layer.sum().item())
        layer_mu = layer_sigma = None
        if layer_keep_cnt >= layer_min_samples:
            vals = wb[keep_layer].float()
            layer_mu    = vals.mean().item()
            layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        layer_max = vals.abs().max().item() if layer_keep_cnt > 0 else None

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]

            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)          # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            # ---- choose (mu, sigma): row -> layer -> global -> orig_row ----
            fallback = None
            if keep_cnt_row >= row_min_samples:
                row_vals = prun_row[keep_row].float()
                mu = row_vals.mean().item()
                sigma = float(row_vals.std(unbiased=False).item() or 1e-8)
            elif (layer_mu is not None) and (layer_sigma is not None):
                mu, sigma = layer_mu, layer_sigma; fallback = "layer"; summary["used_fallback_layer"] += 1
            elif (global_mean is not None) and (global_std is not None):
                mu, sigma = global_mean, global_std; fallback = "global"; summary["used_fallback_global"] += 1
            else:
                alt = orig_row[mask_orig_nz]
                if alt.numel() == 0: continue
                mu = alt.float().mean().item()
                sigma = float(alt.float().std(unbiased=False).item() or 1e-8)
                fallback = "orig_row"; summary["used_fallback_orig_row"] += 1

            # ---- compute row-wise max|.| threshold for hard constraint ----
            if keep_cnt_row > 0:
                row_max = float(prun_row[keep_row].abs().max().item())
            elif layer_max is not None:
                row_max = float(layer_max)
            elif global_max is not None:
                row_max = float(global_max)
            else:
                # fall back to orig row max if available, else 0
                row_max = float(orig_row[mask_orig_nz].abs().max().item()) if mask_orig_nz.any() else 0.0

            margin = max(gt_abs_margin, row_max * gt_rel_margin)
            hard_thresh = row_max + margin  # ensure strict '>' in float32

            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu                         # N(mu, sigma^2)
            mags = samples.abs()
            # enforce |new| > hard_thresh
            mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            # signs: align to orig where orig!=0, else keep sampled sign
            sign_orig = torch.sign(orig_row[pruned_mask_row])   # CPU {-1,0,+1}
            sign_samp = torch.sign(samples)      # CPU {-1,0,+1} (0 rare)

            sign_use  = torch.where(sign_orig == 0, sign_samp, sign_orig)

            new_vals = mags * sign_use                           # CPU values after constraint + sign align

            # write back to device
            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}




@torch.no_grad()
def restore_pruned_weights_rowwise_strict_align_sign_with_3_modules(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    unet_real_orig: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    Max_processing=True,
    Ave_processing=True,
    seed: Optional[int] = 123,
    align_prob: float = 1.0,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())
    c_params = dict(unet_real_orig.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "used_fallback_orig_row": 0,
    }
    per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]                     # will be modified in-place on its device
        wb = pb.detach().cpu()
        wc = c_params[name].detach().cpu()

        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        # layer-level fallbacks (μ/σ) & layer max|.| for the hard constraint
        keep_layer = wb.abs() > eps_pruned
        layer_keep_cnt = int(keep_layer.sum().item())
        layer_mu = layer_sigma = None
        if layer_keep_cnt >= layer_min_samples:
            vals = wb[keep_layer].float()
            layer_mu    = vals.mean().item()
            layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        layer_max = vals.abs().max().item() if layer_keep_cnt > 0 else None

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]
            real_orig_row = wc[r]

            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)          # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            # ---- choose (mu, sigma): row -> layer -> global -> orig_row ----
            fallback = None
            if keep_cnt_row >= row_min_samples:
                row_vals = prun_row[keep_row].float()
                mu = row_vals.mean().item()
                sigma = float(row_vals.std(unbiased=False).item() or 1e-8)
            elif (layer_mu is not None) and (layer_sigma is not None):
                mu, sigma = layer_mu, layer_sigma; fallback = "layer"; summary["used_fallback_layer"] += 1
            elif (global_mean is not None) and (global_std is not None):
                mu, sigma = global_mean, global_std; fallback = "global"; summary["used_fallback_global"] += 1
            else:
                alt = orig_row[mask_orig_nz]
                if alt.numel() == 0: continue
                mu = alt.float().mean().item()
                sigma = float(alt.float().std(unbiased=False).item() or 1e-8)
                fallback = "orig_row"; summary["used_fallback_orig_row"] += 1

            # ---- compute row-wise max|.| threshold for hard constraint ----
            if keep_cnt_row > 0:
                row_max = float(prun_row[keep_row].abs().max().item())
            elif layer_max is not None:
                row_max = float(layer_max)
            elif global_max is not None:
                row_max = float(global_max)
            else:
                # fall back to orig row max if available, else 0
                row_max = float(orig_row[mask_orig_nz].abs().max().item()) if mask_orig_nz.any() else 0.0

            margin = max(gt_abs_margin, row_max * gt_rel_margin)
            hard_thresh = row_max + margin  # ensure strict '>' in float32

            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu                         # N(mu, sigma^2)
            mags = samples.abs()
            # enforce |new| > hard_thresh

            if Max_processing:
                row_max = float(prun_row[keep_row].abs().max().item())
                margin = max(gt_abs_margin, row_max * gt_rel_margin)
                hard_thresh = row_max + margin
                mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            if Ave_processing:
                row_mean = float(prun_row[keep_row].abs().mean().item())
                mags = row_mean

            # mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            # signs: align to orig where orig!=0, else keep sampled sign
            sign_orig = torch.sign(orig_row[pruned_mask_row])  # CPU {-1,0,+1}
            sign_samp = torch.sign(samples)
            sign_real_orig = torch.sign(real_orig_row[pruned_mask_row])# CPU {-1,0,+1} (0 rare)

            sign_use  = torch.where(sign_orig == 0, sign_samp, sign_orig)
            diff_mask = sign_use != sign_real_orig
            final_sign = torch.where(diff_mask, sign_samp, sign_use)

            new_vals = mags * final_sign                          # CPU values after constraint + sign align

            # write back to device
            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}


@torch.no_grad()
def restore_pruned_weights_rowwise_strict_align_sign_with_iter_model(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    # unet_real_orig: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    Max_processing=True,
    Ave_processing=True,
    see: Optional[int] = 123,
    align_prob: float = 1.0,
) -> Dict[str, Any]:
    """
    Row-wise restore with hard magnitude constraint:
      For each pruned (row, col), |new_w| > max_abs(nonzero_in_this_row_of_pruned).
    Signs are aligned to orig at the same positions; if orig==0, keep sampled sign.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())
    # c_params = dict(unet_real_orig.named_parameters())

    # name consistency
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    # global fallback stats (μ/σ & max|.|)
    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "used_fallback_orig_row": 0,
    }
    per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]                     # will be modified in-place on its device
        wb = pb.detach().cpu()
        # wc = c_params[name].detach().cpu()

        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device = pb.device
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0

        # layer-level fallbacks (μ/σ) & layer max|.| for the hard constraint
        keep_layer = wb.abs() > eps_pruned
        layer_keep_cnt = int(keep_layer.sum().item())
        layer_mu = layer_sigma = None
        if layer_keep_cnt >= layer_min_samples:
            vals = wb[keep_layer].float()
            layer_mu    = vals.mean().item()
            layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        layer_max = vals.abs().max().item() if layer_keep_cnt > 0 else None

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]
            # real_orig_row = wc[r]

            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)          # bool [in_features]
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0: continue

            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())

            row_vals = prun_row[keep_row].float()
            mu = row_vals.mean().item()
            sigma = float(row_vals.std(unbiased=False).item() or 1e-8)
            # ---- sample values for this row's pruned positions ----
            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu                         # N(mu, sigma^2)
            mags = samples.abs()
            # enforce |new| > hard_thresh

            if Max_processing:
                row_max = float(prun_row[keep_row].abs().max().item())
                margin = max(gt_abs_margin, row_max * gt_rel_margin)
                hard_thresh = row_max + margin
                mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            if Ave_processing:
                row_mean = float(prun_row[keep_row].abs().mean().item())
                mags = row_mean

            # mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            # signs: align to orig where orig!=0, else keep sampled sign
            sign_orig = torch.sign(orig_row[pruned_mask_row])  # CPU {-1,0,+1}
            sign_samp = torch.sign(samples)
            # sign_real_orig = torch.sign(real_orig_row[pruned_mask_row])# CPU {-1,0,+1} (0 rare)

            sign_use  = torch.where(sign_orig == 0, sign_samp, sign_orig)
            # diff_mask = sign_use != sign_real_orig
            # final_sign = torch.where(diff_mask, sign_samp, sign_use)

            new_vals = mags * sign_use                         # CPU values after constraint + sign align

            # write back to device
            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {"rows_filled": affected_rows,
                               "filled_positions": filled_this_param,
                               "shape": tuple(wb.shape)}
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions"}

    return {"summary": summary, "per_param": per_param}



@torch.no_grad()
def find_pruned_positions(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps: float = 0.0,
    return_indices: bool = False,
    max_indices_per_param: int = 10000,
) -> Dict[str, Any]:
    """
    返回：
      {
        "per_param": {
            <param_name>: {
                "shape": tuple,
                "numel": int,
                "pruned_count": int,
                "pruned_ratio": float,
                "zeros_in_orig": int,
                "zeros_in_pruned": int,
                # 可选:
                # "pruned_flat_indices": np.ndarray (最多 max_indices_per_param 个)
                # "indices_truncated": bool
            },
            ...
        },
        "summary": {
            "total_elems": int,
            "total_pruned": int,
            "global_pruned_ratio": float,
            "num_params": int
        }
      }
    """
    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    shared = sorted(set(a_params.keys()) & set(b_params.keys()))
    per_param: Dict[str, Any] = {}

    total_elems = 0
    total_pruned = 0

    for name in shared:
        wa = a_params[name].detach().cpu()
        wb = b_params[name].detach().cpu()

        if wa.shape != wb.shape:
            per_param[name] = {
                "shape_mismatch": True,
                "a_shape": tuple(wa.shape),
                "b_shape": tuple(wb.shape),
            }
            continue

        mask_orig_nonzero = wa.abs() > eps
        mask_pruned_zero  = wb.abs() <= eps
        pruned_mask = mask_orig_nonzero & mask_pruned_zero

        numel = wa.numel()
        pruned_cnt = int(pruned_mask.sum().item())
        total_elems += numel
        total_pruned += pruned_cnt

        zeros_a = int((wa.abs() <= eps).sum().item())
        zeros_b = int((wb.abs() <= eps).sum().item())

        info: Dict[str, Any] = {
            "shape": tuple(wa.shape),
            "numel": numel,
            "pruned_count": pruned_cnt,
            "pruned_ratio": (pruned_cnt / numel) if numel else 0.0,
            "zeros_in_orig": zeros_a,
            "zeros_in_pruned": zeros_b,
        }

        if return_indices and pruned_cnt > 0:
            flat_idx = torch.nonzero(pruned_mask.flatten(), as_tuple=False).flatten()
            truncated = False
            if flat_idx.numel() > max_indices_per_param:
                flat_idx = flat_idx[:max_indices_per_param]
                truncated = True
            info["pruned_flat_indices"] = flat_idx.numpy()
            info["indices_truncated"] = truncated

        per_param[name] = info

    result = {
        "per_param": per_param,
        "summary": {
            "total_elems": total_elems,
            "total_pruned": total_pruned,
            "global_pruned_ratio": (total_pruned / total_elems) if total_elems else 0.0,
            "num_params": len(shared),
        }
    }
    return result



def get_pruned_param_names(report, min_pruned_count: int = 1):
    """
    从 find_pruned_positions 的 report 中，拿到所有被剪枝的“参数名”列表。
    规则：pruned_count >= min_pruned_count
    """
    pruned_params = []
    for name, info in report["per_param"].items():
        if info.get("shape_mismatch"):
            continue
        if info.get("pruned_count", 0) >= min_pruned_count:
            pruned_params.append(name)
    return sorted(pruned_params)


def _parent_module_name(param_name: str):
    """
    把 'xxx.weight' 或 'xxx.bias' 这样的参数名映射到父 module 名 'xxx'。
    如果没有 .weight/.bias 后缀，就返回去掉最后一段的父级（尽量保守）。
    """
    if param_name.endswith(".weight") or param_name.endswith(".bias"):
        return param_name.rsplit(".", 1)[0]
    parts = param_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else param_name


def get_pruned_modules(report, min_pruned_count: int = 1):


    per_mod = {}
    for name, info in report["per_param"].items():
        if info.get("shape_mismatch"):
            continue
        pruned_cnt = int(info.get("pruned_count", 0))
        if pruned_cnt < min_pruned_count:
            continue

        mod = _parent_module_name(name)
        entry = per_mod.setdefault(mod, {"pruned_count_sum": 0, "numel_sum": 0, "params": []})
        entry["pruned_count_sum"] += pruned_cnt
        entry["numel_sum"] += int(info.get("numel", 0))
        entry["params"].append(name)


    results = []
    for mod, d in per_mod.items():
        numel_sum = max(d["numel_sum"], 1)
        pr_ratio = d["pruned_count_sum"] / numel_sum
        results.append((mod, {
            "pruned_count_sum": d["pruned_count_sum"],
            "numel_sum": numel_sum,
            "pruned_ratio": pr_ratio,
            "params": sorted(d["params"]),
        }))


    results.sort(key=lambda x: (x[1]["pruned_ratio"], x[1]["pruned_count_sum"]), reverse=True)
    return results




@torch.no_grad()
def restore_pruned_weights_rowwise_strict_no_signalign(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    eps_orig: float = 0.0,
    eps_pruned: float = 0.0,
    ffn2_only: bool = True,
    row_min_samples: int = 16,
    layer_min_samples: int = 128,
    gt_rel_margin: float = 1e-6,
    gt_abs_margin: float = 1e-12,
    seed: Optional[int] = 123,
) -> Dict[str, Any]:




    g = None
    if seed is not None:
        g = torch.Generator(device="cpu"); g.manual_seed(seed)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = set(b_params.keys()) - set(a_params.keys())
        missing_in_b = set(a_params.keys()) - set(b_params.keys())
        raise ValueError(
            "参数名不一致：\n"
            f"  只在原模型缺失: {sorted(missing_in_a)}\n"
            f"  只在剪枝模型缺失: {sorted(missing_in_b)}"
        )

    def _is_target_weight(n: str) -> bool:
        if ffn2_only and (".ff." not in n or ".net.2." not in n): return False
        return n.endswith(".weight")

    target_names = [n for n in sorted(a_params.keys()) if _is_target_weight(n)]

    global_pool = []
    for n in target_names:
        wb = b_params[n].detach().cpu()
        if wb.ndim != 2: continue
        keep = wb.abs() > eps_pruned
        if keep.any(): global_pool.append(wb[keep].float())
    global_mean = global_std = global_max = None
    if global_pool:
        pooled = torch.cat(global_pool)
        global_mean = pooled.mean().item()
        global_std  = float(pooled.std(unbiased=False).item() or 1e-8)
        global_max  = pooled.abs().max().item()

    summary = {
        "affected_params": 0,
        "affected_rows": 0,
        "filled_positions": 0,
        "used_fallback_layer": 0,
        "used_fallback_global": 0,
        "skipped_rows_no_stats": 0,
    }
    per_param: Dict[str, Any] = {}

    for name in target_names:
        wa = a_params[name].detach().cpu()
        pb = b_params[name]
        wb = pb.detach().cpu()

        if wa.ndim != 2 or wb.ndim != 2 or wa.shape != wb.shape:
            per_param[name] = {"skipped": True, "reason": "not_2d_or_shape_mismatch"}
            continue

        device, dtype = pb.device, pb.dtype
        out_features, in_features = wb.shape
        filled_this_param = 0
        affected_rows = 0


        keep_layer = wb.abs() > eps_pruned
        layer_keep_cnt = int(keep_layer.sum().item())
        layer_mu = layer_sigma = None
        if layer_keep_cnt >= layer_min_samples:
            vals = wb[keep_layer].float()
            layer_mu = vals.mean().item()
            layer_sigma = float(vals.std(unbiased=False).item() or 1e-8)
        layer_max = (vals.abs().max().item() if layer_keep_cnt > 0 else None)

        for r in range(out_features):
            orig_row = wa[r]
            prun_row = wb[r]


            mask_orig_nz = orig_row.abs() > eps_orig
            mask_prun_z  = prun_row.abs() <= eps_pruned
            pruned_mask_row = (mask_orig_nz & mask_prun_z)
            pruned_cnt_row = int(pruned_mask_row.sum().item())
            if pruned_cnt_row == 0:
                continue


            keep_row = prun_row.abs() > eps_pruned
            keep_cnt_row = int(keep_row.sum().item())


            fallback = None
            if keep_cnt_row >= row_min_samples:
                row_vals = prun_row[keep_row].float()
                mu = row_vals.mean().item()
                sigma = float(row_vals.std(unbiased=False).item() or 1e-8)
            elif (layer_mu is not None) and (layer_sigma is not None):
                mu, sigma = layer_mu, layer_sigma
                fallback = "layer"; summary["used_fallback_layer"] += 1
            elif (global_mean is not None) and (global_std is not None):
                mu, sigma = global_mean, global_std
                fallback = "global"; summary["used_fallback_global"] += 1
            else:

                summary["skipped_rows_no_stats"] += 1
                continue


            if keep_cnt_row > 0:
                row_max = float(prun_row[keep_row].abs().max().item())
            elif layer_max is not None:
                row_max = float(layer_max)
            elif global_max is not None:
                row_max = float(global_max)
            else:
                row_max = 0.0

            margin = max(gt_abs_margin, row_max * gt_rel_margin)
            hard_thresh = row_max + margin


            noise = torch.randn((pruned_cnt_row,), generator=g) if g is not None else torch.randn((pruned_cnt_row,))
            samples = noise * sigma + mu
            mags = samples.abs()
            mags = torch.maximum(mags, torch.full_like(mags, hard_thresh))

            sign_samp = torch.sign(samples)

            sign_samp[sign_samp == 0] = 1.0

            new_vals = mags * sign_samp


            col_idx = torch.nonzero(pruned_mask_row, as_tuple=False).flatten().to(device)
            pb_row = pb[r]
            pb_row[col_idx] = new_vals.to(device=pb_row.device, dtype=pb_row.dtype)

            filled_this_param += pruned_cnt_row
            affected_rows += 1

        if filled_this_param > 0:
            summary["affected_params"] += 1
            summary["affected_rows"] += affected_rows
            summary["filled_positions"] += filled_this_param
            per_param[name] = {
                "rows_filled": affected_rows,
                "filled_positions": filled_this_param,
                "shape": tuple(wb.shape),
            }
        else:
            per_param[name] = {"skipped": True, "reason": "no_pruned_positions_or_no_stats"}

    return {"summary": summary, "per_param": per_param}


def quantize_selected_modules(unet, target_modules):
    for name in target_modules:
        module = unet.get_submodule(name)
        print(f"Quantizing {name}: {type(module)}")
        quantized = quantize_linear_to_4bit(module)
        replace_module(unet, name, quantized)


def replace_module(model: nn.Module, module_path: str, new_module: nn.Module):
    """根据路径把 model 中的某个模块替换为 new_module"""
    parent_name = ".".join(module_path.split(".")[:-1])
    child_name = module_path.split(".")[-1]

    parent_module = model.get_submodule(parent_name) if parent_name else model
    setattr(parent_module, child_name, new_module)



@torch.no_grad()
def quant_model(unet_pruned: torch.nn.Module) -> Dict[str, Any]:
    b_params = dict(unet_pruned.named_parameters())

    def is_ffn2_weight(n: str) -> bool:
        return (".ff." in n) and (".net.2." in n) and n.endswith(".weight")



    target_names = [n for n in sorted(b_params.keys()) if is_ffn2_weight(n)]

    for name in target_names:
        pb = b_params[name]
        device = pb.device
        wb = pb.detach().cpu()

        qbit = 4
        qmax = 2 ** (qbit - 1) - 1  # 7
        qmin = -2 ** (qbit - 1)  # -8
        scale = wb.abs().amax(dim=1, keepdim=True) / qmax
        w_int = torch.floor(wb / scale + 0.5).clamp(qmin, qmax)
        w_dequant = (w_int * scale).to(device)

        pb.data.copy_(w_dequant)



        # module_name = name.replace(".weight", "")
        # module = unet_pruned.get_submodule(module_name)







# def quantize_model_4bit(module):
#     """
#     """
#     if isinstance(module, nn.Linear):
#         quantized = bnb.nn.Linear4bit(
#             in_features=module.in_features,
#             out_features=module.out_features,
#             bias=module.bias is not None,
#         )
#         quantized.weight.data = module.weight.data.clone()
#         if module.bias is not None:
#             quantized.bias.data = module.bias.data.clone()
#         return quantized
#     else:
#         return module


# 'Recover_via_CSV_sign_only_pruned_model_participated'

def main_csv_recover(sign_seeds, model_orig, model_pruned, unlearn_ckpt, csv_folder, mag_process):

    magnitude_processing = mag_process

    if magnitude_processing == 'Ave':
        Strictly_Align = False
        Max_process = False
        Ave_process = True

    elif magnitude_processing == 'Max':
        Strictly_Align = False
        Max_process = True
        Ave_process = False

    elif magnitude_processing == 'Sample':
        Strictly_Align = False
        Max_process = False
        Ave_process = False

    elif magnitude_processing == 'Align':
        Strictly_Align = True
        Max_process = False
        Ave_process = False
    else:
        raise ValueError(f"Unsupported magnitude_processing: {magnitude_processing}")

    files = glob.glob(os.path.join(csv_folder, "*/"))
    csv_files = [os.path.basename(os.path.normpath(d)) for d in files]
    # csv_files = ['GPU_200it_1e5cv_rankNone_keep_row_top_0.8']

    csv_files = [f"GPU_500it_1e5cv_rankNone_ft64_Top{args.top_ratio}"]
    for csv_name in csv_files:
        weight_path = os.path.join(csv_folder, csv_name)
        save_name = f'Recover_via_{csv_name.split("GPU_")[1]}_Sign_{magnitude_processing}_processing'
        for sign in sign_seeds:
            # image_save_path = os.path.join(
            #     f'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/{args.target}/after_recover_results', f'{save_name}')
            # # print(f"Running Mag_seed={mag}")
            # os.makedirs(image_save_path, exist_ok=True)


            full_state = torch.load(unlearn_ckpt, map_location="cpu")
            model_pruned.unet.load_state_dict(full_state, strict=False)


            model_pruned = model_pruned.to(args.gpu)
            # model_orig = model_orig.to(args.gpu)


            unet_pruned = model_pruned.unet
            # unet_orig = model_orig.unet

            # restore_pruned_weights_rowwise_align_sign_with_activation_sort(unet_orig=unet_orig, unet_pruned=unet_pruned, Mag_seed=sign, Max_processing=Max_process, Ave_processing=Ave_process, bottom_keep=False, opposite_set_zero=True, align_prob=0.6)

            restore_pruned_weights_with_csv(unet_pruned=unet_pruned, path=weight_path, Mag_seed=sign, Max_processing=Max_process,
                                            Ave_processing=Ave_process, CSV_Align=Strictly_Align)

            # iter = 0

            # for ann_target in target_prompts:
            #     if iter >= 2 and args.dbg:
            #         break
            #     print("text: ", ann_target)
            #
            #     torch.manual_seed(args.seed)
            #     np.random.seed(args.seed)
            #     out_base = model_orig(ann_target).images[0]
            #
            #     torch.manual_seed(args.seed)
            #     np.random.seed(args.seed)
            #     out_target = model_pruned(ann_target, safety_checker=safety_checker.StableDiffusionSafetyChecker).images[0]
            #
            #     out_base = out_base.resize((256, 256))
            #     out_target = out_target.resize((256, 256))
            #
            #     new_im = Image.new('RGB', (530, 290))
            #     new_im.paste(out_base, (0, 40))
            #     new_im.paste(out_target, (275, 40))
            #
            #     draw = ImageDraw.Draw(new_im)
            #     draw.text((80, 15), ann_target, (255, 255, 255))
            #     draw.text((350, 15), 'Recovery', (255, 255, 255))

                # print("Saving image in: ", os.path.join(image_save_path, f'img_{iter}_{ann_target}.jpg'))

                # new_im.save(os.path.join(image_save_path, f'img_{iter}_{ann_target}.jpg'))
                # iter += 1
            # model_save_path = os.path.join(
            #     f'/scratch/cz06540/models_for_ablation_study/{args.target}/{save_name}', 'model')

            model_save_path = os.path.join(
                f'/scratch/cz06540/defense_rebuttal_csv/recovered_model/{args.target}/{save_name}', 'model')

            os.makedirs(model_save_path, exist_ok=True)

            checkpoint_path = f"/scratch/cz06540/defense_rebuttal_csv/recovered_model/{args.target}/{save_name}/model/filled_with_mag_seed{sign}.pt"
            state_cpu = {k: v.detach().cpu() for k, v in model_pruned.unet.state_dict().items()}
            torch.save(state_cpu, checkpoint_path)
            print("saved to:", checkpoint_path)








args = Config('../configs/wanda_config.yaml')
cmd_args = input_args()
# iterate over the args and update the config
for key, value in vars(cmd_args).items():
    if value is not None:
        print(f"Updating {key} with {value}")
        setattr(args, key, value)

args.configure()

print("Arguments: ", args.__dict__)
base_prompts, target_prompts = get_prompts(args)
print("Base prompts: ", base_prompts)
print("Target prompts: ", target_prompts)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# Model
model_orig, num_layers, replace_fn = get_sd_model(args)
args.replace_fn = replace_fn
model_orig = model_orig.to(args.gpu)
model_orig.unet.eval()
model_pruned = copy.deepcopy(model_orig).to('cpu')
model_pruned_2 = copy.deepcopy(model_orig).to('cpu')

# unlearn_ckpt = f"/home/cz06540/concept-prune/wanda/results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/{args.target}/checkpoints/skill_ratio_0.02_accumlate_timesteps_10_threshold0.pt"
unlearn_ckpt = f'/scratch/cz06540/defense_rebuttal/{args.target}/defense_filled_repruned_via_global_0.02_bottom.pt'


full_state = torch.load(unlearn_ckpt, map_location="cpu")
model_pruned.unet.load_state_dict(full_state, strict=False)
model_pruned = model_pruned.to(args.gpu)

unet_orig  = model_orig.unet
# unet_pruned = model_pruned.unet
# unet_pruned_2 = model_pruned_2.unet

# report = restore_ffn2_by_sign_alignment(unet_orig=unet_orig, unet_pruned=unet_pruned)
# softimpute_unet_ffn2(unet_pruned_2)

mag_seeds_list = [2021, 2022, 2023, 2024, 2025, 2026]
sign_seeds_list = [2021]

# quant_model(unet_pruned)
# restore_pruned_weights_rowwise_strict_align_sign()
# restore_pruned_weights_rowwise_dynamic_align_sign(unet_orig=unet_orig, unet_pruned=unet_pruned, Max_processing=False, Ave_processing=False)
# restore_pruned_weights_rowwise_align_magnitudes(unet_orig=unet_orig, unet_pruned=unet_pruned)
main_csv_recover(mag_process=args.magnitude_process, sign_seeds=sign_seeds_list, model_orig=model_orig, model_pruned=model_pruned, unlearn_ckpt=unlearn_ckpt, csv_folder=args.csv_folder)

# model_save_path = os.path.join(f'/scratch/cz06540/Pretrain_Signs_Random_Weights/{args.target}')
# #
# os.makedirs(model_save_path, exist_ok=True)
# #
# checkpoint_path = f"/scratch/cz06540/Pretrain_Signs_Random_Weights/{args.target}/filled_with_mag_seed2021.pt"
# state_cpu = {k: v.detach().cpu() for k, v in model_pruned.unet.state_dict().items()}
# torch.save(state_cpu, checkpoint_path)
# print("saved to:", checkpoint_path)



# for mag in mag_seeds:
# for sign in sign_seeds:
#     save_name = 'Recover_via_CSV_sign_only_pruned_model_participated'
#     image_save_path = os.path.join(
#         f'results_Ci/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball',
#         '0.02sparsity_10_timestep_0thre',f'{save_name}','Image',
#         f'Generated_Image_mag_seeds{sign}', )
#     # print(f"Running Mag_seed={mag}")
#     os.makedirs(image_save_path, exist_ok=True)
#
#     unet_orig_copy = copy.deepcopy(unet_orig)
#     full_state = torch.load(unlearn_ckpt, map_location="cpu")
#     model_pruned.unet.load_state_dict(full_state, strict=False)
#     model_pruned_2.unet.load_state_dict(full_state, strict=False)
#
#     model_pruned = model_pruned.to(args.gpu)
#     model_pruned_2 = model_pruned_2.to(args.gpu)
#
#
#     unet_pruned = model_pruned.unet
#     unet_pruned_2 = model_pruned_2.unet
#
#     torch.manual_seed(sign)
#     np.random.seed(sign)
#     # softimpute_unet_ffn2(unet_pruned_2)
#
#     restore_pruned_weights_with_csv(unet_pruned=unet_pruned, Mag_seed=sign, Max_processing=False, Ave_processing=False)
#
#     iter = 0
#
#     for ann_target in target_prompts:
#         if iter >= 2 and args.dbg:
#             break
#         print("text: ", ann_target)
#
#
#         torch.manual_seed(args.seed)
#         np.random.seed(args.seed)
#         out_base = model_orig(ann_target).images[0]
#
#         torch.manual_seed(args.seed)
#         np.random.seed(args.seed)
#         out_target = model_pruned(ann_target, safety_checker=safety_checker.StableDiffusionSafetyChecker).images[0]
#
#
#
#
#         out_base = out_base.resize((256, 256))
#         out_target = out_target.resize((256, 256))
#
#
#         new_im = Image.new('RGB', (530, 290))
#         new_im.paste(out_base, (0, 40))
#         new_im.paste(out_target, (275, 40))
#
#         draw = ImageDraw.Draw(new_im)
#         draw.text((80, 15), ann_target, (255, 255, 255))
#         draw.text((350, 15), 'Recovery', (255, 255, 255))
#
#         print("Saving image in: ", os.path.join(image_save_path, f'img_{iter}_{ann_target}.jpg'))
#
#         new_im.save(os.path.join(image_save_path, f'img_{iter}_{ann_target}.jpg'))
#         iter += 1
#     model_save_path = os.path.join('results_Ci/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball',
#             '0.02sparsity_10_timestep_0thre', f'{save_name}', 'model')
#
#     os.makedirs(model_save_path, exist_ok=True)
#
#     checkpoint_path = f"results_Ci/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/0.02sparsity_10_timestep_0thre/{save_name}/model/filled_with_mag_seed{sign}.pt"
#     state_cpu = {k: v.detach().cpu() for k, v in model_pruned.unet.state_dict().items()}
#     torch.save(state_cpu, checkpoint_path)
#     print("saved to:", checkpoint_path)








# iter = 0
# for ann_target in target_prompts:
#     if iter >= 2 and args.dbg:
#         break
#     print("text: ", ann_target)
#
#     # fix seed before running the model
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     out_base = model_orig(ann_target).images[0]
#
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     out_target = model_pruned(ann_target, safety_checker=safety_checker.StableDiffusionSafetyChecker).images[0]
#
#     # stitch the images to keep them side by side
#     out_base = out_base.resize((256, 256))
#     out_target = out_target.resize((256, 256))
#
#     # make bigger image to keep both images side by side with white space in between
#     new_im = Image.new('RGB', (530, 290))
#     new_im.paste(out_base, (0, 40))
#     new_im.paste(out_target, (275, 40))
#     # write the prompt on the image
#     draw = ImageDraw.Draw(new_im)
#     draw.text((80, 15), ann_target, (255, 255, 255))
#     draw.text((350, 15), 'Recovery', (255, 255, 255))
#     print("Saving image in: ", os.path.join(save_path, f'img_{iter}_{ann_target}.jpg'))
#     os.makedirs(save_path, exist_ok=True)
#     new_im.save(os.path.join(save_path, f'img_{iter}_{ann_target}.jpg'))
#     iter += 1
#
# checkpoint_path = "results_Ci/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/filled_with_zhaojun_signs_and_rest_distribution_seed_2021_0.35_Opposite_seed_222/0.02sparsity_10_timestep_0thre/recoverd_model.pt"
#
#
# state_cpu = {k: v.detach().cpu() for k, v in model_pruned.unet.state_dict().items()}
# torch.save(state_cpu, checkpoint_path)
# print("saved to:", checkpoint_path)



