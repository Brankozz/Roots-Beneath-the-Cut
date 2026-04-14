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

from PIL import Image, ImageFilter, ImageDraw

from diffusers.pipelines.stable_diffusion import safety_checker
import matplotlib.pyplot as plt

from utils import get_prompts, Config, get_sd_model
sys.path.append(os.getcwd())
# os.chdir("/home/cz06540/concept-prune/revive")
import seaborn as sns
from neuron_receivers import Wanda
from transformers.models.clip.modeling_clip import CLIPMLP
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
import math
import os
import re
import csv



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








def _sanitize_filename(name: str) -> str:
    """Make a safe filename from a parameter name."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)

@torch.no_grad()
def export_unet_ffn2_params(
    unet_orig: torch.nn.Module,
    unet_pruned: torch.nn.Module,
    out_dir: str,
    eps_pruned: float = 0.0,      # magnitude threshold to treat pruned weights as ~0
    include_bias: bool = True,    # also export bias in FFN second linear layer if present
    save_pruned_indices: bool = True,  # save indices of pruned positions for each param
    flatten: bool = True          # save as a single column CSV (recommended)
) -> str:
    """
    Export parameters ONLY for FFN second linear layers ('.ff.net.2.') from both orig and pruned UNet.
    For each targeted param, write two CSVs (orig/pruned), and optionally the pruned indices CSV.
    Also write a summary.csv with shapes and pruning stats.

    Returns:
        Path to summary.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    dir_orig = os.path.join(out_dir, "orig_csv")
    dir_prun = os.path.join(out_dir, "pruned_csv")
    dir_idx  = os.path.join(out_dir, "pruned_indices")
    os.makedirs(dir_orig, exist_ok=True)
    os.makedirs(dir_prun, exist_ok=True)
    if save_pruned_indices:
        os.makedirs(dir_idx, exist_ok=True)

    a_params = dict(unet_orig.named_parameters())
    b_params = dict(unet_pruned.named_parameters())

    # Strong consistency check
    if set(a_params.keys()) != set(b_params.keys()):
        missing_in_a = sorted(set(b_params.keys()) - set(a_params.keys()))
        missing_in_b = sorted(set(a_params.keys()) - set(b_params.keys()))
        raise ValueError(
            "Parameter names differ between orig and pruned UNet.\n"
            f"Missing in orig: {missing_in_a[:5]}{' ...' if len(missing_in_a) > 5 else ''}\n"
            f"Missing in pruned: {missing_in_b[:5]}{' ...' if len(missing_in_b) > 5 else ''}"
        )

    names = sorted(a_params.keys())

    def _is_ffn2(name: str) -> bool:
        """Match FFN second linear layer in SD/diffusers UNet."""
        if ".ff." not in name or ".net.2." not in name:
            return False
        if name.endswith(".weight"):
            return True
        return include_bias and name.endswith(".bias")

    target_names = [n for n in names if _is_ffn2(n)]

    # Prepare summary
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as fsum:
        writer = csv.writer(fsum)
        writer.writerow([
            "param_name",
            "shape",
            "numel",
            "zeros_in_pruned(@eps)",
            "pruned_ratio(@eps)",
            "orig_csv_path",
            "pruned_csv_path",
            "pruned_indices_csv_path" if save_pruned_indices else "pruned_indices_csv_path_disabled",
        ])

        for name in target_names:
            wa = a_params[name].detach().cpu().numpy()
            wb_t = b_params[name]
            wb = wb_t.detach().cpu().numpy()

            # File paths
            base = _sanitize_filename(name)
            path_a = os.path.join(dir_orig, f"{base}.csv")
            path_b = os.path.join(dir_prun, f"{base}.csv")
            path_idx = os.path.join(dir_idx, f"{base}_pruned_idx.csv") if save_pruned_indices else ""

            # Save weights to CSV (flatten for consistency/readability)
            if flatten:
                np.savetxt(path_a, wa.reshape(-1, 1), delimiter=",")
                np.savetxt(path_b, wb.reshape(-1, 1), delimiter=",")
            else:
                # Keep 2D shape when possible; otherwise fall back to flattened
                if wa.ndim == 2:
                    np.savetxt(path_a, wa, delimiter=",")
                else:
                    np.savetxt(path_a, wa.reshape(-1, 1), delimiter=",")
                if wb.ndim == 2:
                    np.savetxt(path_b, wb, delimiter=",")
                else:
                    np.savetxt(path_b, wb.reshape(-1, 1), delimiter=",")

            # Pruning stats with magnitude threshold eps_pruned
            numel = wa.size
            zeros_in_pruned = int((np.abs(wb) <= eps_pruned).sum())
            pruned_ratio = zeros_in_pruned / max(1, numel)

            if save_pruned_indices:
                pruned_mask = (np.abs(wa) > eps_pruned) & (np.abs(wb) <= eps_pruned)

                if wa.ndim == 2:
                    rc_idx = np.argwhere(pruned_mask)  # shape: (K, 2)
                    np.savetxt(
                        path_idx,
                        rc_idx.astype(np.int64),
                        fmt="%d",
                        delimiter=",",
                        header="row,col",
                        comments=""
                    )
                else:
                    idx = np.nonzero(pruned_mask.reshape(-1))[0].astype(np.int64).reshape(-1, 1)
                    np.savetxt(
                        path_idx,
                        idx,
                        fmt="%d",
                        delimiter=",",
                        header="idx",
                        comments=""
                    )

            writer.writerow([
                name,
                str(tuple(wa.shape)),
                numel,
                zeros_in_pruned,
                f"{pruned_ratio:.8f}",
                path_a,
                path_b,
                path_idx if save_pruned_indices else "",
            ])

        return summary_path





args = Config('../configs/wanda_config.yaml')
cmd_args = input_args()
# iterate over the args and update the config
for key, value in vars(cmd_args).items():
    if value is not None:
        print(f"Updating {key} with {value}")
        setattr(args, key, value)

args.configure()

# print("Arguments: ", args.__dict__)
# base_prompts, target_prompts = get_prompts(args)
# print("Base prompts: ", base_prompts)
# print("Target prompts: ", target_prompts)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# Model
model_orig, num_layers, replace_fn = get_sd_model(args)
args.replace_fn = replace_fn
model_orig = model_orig.to(args.gpu)
model_orig.unet.eval()
model_pruned = copy.deepcopy(model_orig).to('cpu')

# unlearn_ckpt = f'/home/cz06540/concept-prune/wanda/results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/{args.target}/checkpoints/skill_ratio_0.02_accumlate_timesteps_10_threshold0.pt'
unlearn_ckpt = f'/scratch/cz06540/defense_rebuttal/{args.target}/defense_filled_repruned_via_global_0.02_bottom.pt'

full_state = torch.load(unlearn_ckpt, map_location="cpu")


model_pruned.unet.load_state_dict(full_state, strict=False)
model_pruned = model_pruned.to(args.gpu)
# save_path = os.path.join('results_Ci/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball', 'filled_based_on_global_origin', '0.02sparsity_0.2select_ration')


unet_a = model_orig.unet
unet_b = model_pruned.unet



# out_dir = f"/scratch/cz06540/concept-prune-csv/{args.target}"
out_dir = f"/scratch/cz06540/defense_rebuttal_csv/{args.target}"

summary_csv = export_unet_ffn2_params(
    unet_orig=model_orig.unet,
    unet_pruned=model_pruned.unet,
    out_dir=out_dir,
    eps_pruned=0.0,
    include_bias=False,
    save_pruned_indices=True,
    flatten=False
)

print("Summary saved to:", summary_csv)
