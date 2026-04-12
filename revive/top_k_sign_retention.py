import copy
import os
import json

import functorch.dim
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

from utils import get_prompts, Config, get_sd_model
os.chdir("/home/cz06540/concept-prune/wanda")
import seaborn as sns
from neuron_receivers import Wanda
from transformers.models.clip.modeling_clip import CLIPMLP
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import math
import pandas as pd
import glob



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
    parser.add_argument('--top_ratio', type=float, default=0.8)
    return parser.parse_args()


args = Config('../configs/wanda_config.yaml')
cmd_args = input_args()

for key, value in vars(cmd_args).items():
    if value is not None:
        print(f"Updating {key} with {value}")
        setattr(args, key, value)

args.configure()



model_orig, num_layers, replace_fn = get_sd_model(args)
args.replace_fn = replace_fn
model_orig = model_orig.to(args.gpu)
model_orig.unet.eval()
model_pruned = copy.deepcopy(model_orig).to('cpu')





# recovered_path = f"/scratch/cz06540/concept-prune-csv/{args.target}/output/GPU_500it_1e5cv_rankNone_ft64"
# pruned_path = f"/scratch/cz06540/concept-prune-csv/{args.target}/pruned_csv"

recovered_path = f"/scratch/cz06540/defense_rebuttal_csv/{args.target}/output/GPU_500it_1e5cv_rankNone_ft64"
pruned_path = f"/scratch/cz06540/defense_rebuttal_csv/{args.target}/pruned_csv"

# full_state = torch.load(unlearn_ckpt, map_location="cpu")
# model_pruned.unet.load_state_dict(full_state, strict=False)
# model_pruned = model_pruned.to(args.gpu)

# unet_pruned = model_pruned.unet

filenames = glob.glob(os.path.join(recovered_path, "*.csv"))
filenames = [os.path.basename(f) for f in filenames]

# params = dict(unet_pruned.named_parameters())





# filenames = glob.glob(os.path.join(path, "*.csv"))
# filenames = [os.path.basename(f) for f in filenames]

# params = dict(unet_pruned.named_parameters())

for filename in filenames:
    pruned_name = filename.split("SoftImpute_")[-1]

    recovred_weight = os.path.join(recovered_path, filename)

    pruned_weight = os.path.join(pruned_path, pruned_name)

    weight_recovered = pd.read_csv(recovred_weight, header=None)
    weight_pruned = pd.read_csv(pruned_weight, header=None)

    weight_tensor_recovered = torch.tensor(weight_recovered.values, dtype=torch.float32).to('cpu')
    print(weight_tensor_recovered.shape)
    weight_tensor_pruend = torch.tensor(weight_pruned.values, dtype=torch.float32).to('cpu')
    print(weight_tensor_pruend.shape)
    # weight_pruned = params[weight_name].to('cpu')
    # print(weight_pruned.shape, flush=True)

    zero_mask = weight_tensor_pruend.abs() <= 0.0

    # prune_map = weight_tensor[zero_mask]
    # print(prune_map.numel(), flush=True)

    prune_map = weight_tensor_recovered[zero_mask]

    prune_map_abs = prune_map.abs()

    # bs_select = prune_map.abs()a

    # nonzero_mask = abs_select > 0.0
    # print(nonzero_mask.shape, flush=True)

    # nonzero_vals = abs_select[prune_map]
    # print(nonzero_vals.numel(), flush=True)

    if prune_map_abs.numel() > 0:
        k = int((1 - args.top_ratio) * prune_map_abs.numel())
        if k > 0:
            threshold = torch.topk(prune_map_abs, k, largest=False).values.max()

            zero_prune_selected = prune_map_abs <= threshold

            selected_copy = prune_map.clone()
            selected_copy[zero_prune_selected] = 0

            weight_tensor_processed = weight_tensor_recovered.clone()
            weight_tensor_processed[zero_mask] = selected_copy

            # parent_dir = f"/scratch/cz06540/concept-prune-csv/{args.target}/top_ratio_output/GPU_500it_1e5cv_rankNone_ft64_Top{args.top_ratio}"

            parent_dir = f"/scratch/cz06540/defense_rebuttal_csv/{args.target}/top_ratio_output/GPU_500it_1e5cv_rankNone_ft64_Top{args.top_ratio}"


            os.makedirs(parent_dir, exist_ok=True)

            save_dir = os.path.join(parent_dir, f'{os.path.splitext(pruned_name)[0]}.csv')

            np.savetxt(save_dir, weight_tensor_processed.numpy(), delimiter=",")



























































