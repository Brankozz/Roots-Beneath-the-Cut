# Roots Beneath the Cut

Code for the paper - **Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models**

[![arXiv](https://img.shields.io/badge/arXiv-2603.06640-b31b1b.svg)](https://arxiv.org/abs/2603.06640)

## Introduction

<p align="center">
<img src="Image/Picture_hacker.png" width="50%">
</p>

Pruning-based unlearning has recently emerged as a fast, training-free, and data-independent approach to remove undesired concepts from diffusion models. It promises high efficiency and robustness, offering an attractive alternative to traditional fine-tuning or editing-based unlearning. However, in this paper we uncover a hidden danger behind this promising paradigm. We find that the locations of pruned weights, typically set to zero during unlearning, can act as side-channel signals that leak critical information about the erased concepts. To verify this vulnerability, we design a novel attack framework capable of reviving erased concepts from pruned diffusion models in a fully data-free and training-free manner. Our experiments confirm that pruning-based unlearning is not inherently secure, as erased concepts can be effectively revived without any additional data or retraining. Extensive experiments on diffusion-based unlearning based on concept related weights lead to the conclusion: once the critical concept-related weights in diffusion models are identified, our method can effectively recover the original concept regardless of how the weights are manipulated. Finally, we explore potential defense strategies and advocate safer pruning mechanisms that conceal pruning locations while preserving unlearning effectiveness, providing practical insights for designing more secure pruning-based unlearning frameworks.

## Environment Setup

Create Environment from the `environment.yml` file.

`cd env`

`conda env create -f environment.yml`

`conda activate concept_revival`

## Obtain the Unlearned Model

<p align="center">
<img src="Image/unlearn.png" width="50%">
</p>

To obtain the unlearned model for a concept `<target>`, run the following-

`python revive.wanda --target="$target" --skill_ratio 0.02`

`python revive.save_union_over_time --target="$target" --timesteps 10 --skill_ratio 0.02`

`<target>` is the concept that we want to erase. Replace `<target>` with any of -

&nbsp; 1. Artist Styles - `Van Gogh, Monet, Pablo Picasso, Da Vinci, Salvador Dali`. Example - base prompt = `a cat` and target prompt = `a cat in the style of Van Gogh`

&nbsp; 2. Nudity - `naked`. Example - base prompt = `a photo of a man` and target prompt = `a photo of a naked man`

&nbsp; 3. Objects (Imagenette classes) - `golf ball, parachute, church, french horn, chain saw, gas pump, candle, mountain bike, racket, school bus, spider web, starfish`.

&nbsp; Example - base prompt = `a room` and target prompt = `a parachute in a room`

The argument `skill_ratio` denotes the sparsity level which defines the top-k% neurons considered for WANDA pruning. This command saves skilled neurons discovered for every timestep and layer in a different .pkl file as a sparse matrix. We recommend using `0.02` for all object and artist style tasks, and `0.01` for the nudity task.

**Note: You can also use your own unlearned model obtained by detecting concept-related weights, as long as the locations of the concept-related weights remain identifiable.**

## Matrix Completion

<p align="center">
<img src="Image/matrix_completion.png" width="50%">
</p>

To recover the pruned matrix, run the following-

**(Note: You need to set the concept-related weights to zero first if you are using your own unlearned model.)**

`python -m revive.read_weights --target="$target"`

`python -m revive.matrix_completion_lterative_Soft-Thresholded_SVD_gpu --target="$target"`

## Top-K Sign Retention

<p align="center">
<img src="Image/top_k.png" width="25%">
</p>

To preserve the signs with Top-k magnitudes, run the following-

`python -m revive.top_k_sign_retention --target="$target" --top_ratio="$top_ratio"`

You could try `top_ratio` from `(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)`, as the recovery performance varies across different concepts for this parameter. All experiments in the paper use `0.6`.

## Neuron Max Scaling

To maximize the magnitudes of remained signs, run the following-

`python revive.neuron_max_scaling --target="$target" --csv_folder "path"`

## Evaluation

### Artist Styles

To evaluate artist style erasure for `Van Gogh, Monet, Pablo Picasso, Da Vinci, Salvador Dali`, run

`python benchmarking/artist_erasure.py --target <target> --baseline concept-prune --ckpt_name <path to checkpoint>`

We created a dataset of 50 prompts using ChatGPT for different artists such that each prompt contains the painting name along with the name of the artist. These prompts are available in `datasets/`. The script saves images and a json files with CLIP metric reported in the paper in the `results/` folder.

### Nudity

To evaluate nudity erasure on the I2P dataset, run

`python benchmarking/nudity_eval.py --eval_dataset i2p --baseline 'concept-prune' --gpu 0 --ckpt_name <path to checkpoint>`

To run on black-box adversarial prompt datasets, [MMA](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf) and [Ring-A-Bell](https://arxiv.org/abs/2310.10012), replace `i2p` with `mma` and `ring-a-bell` respectively.

We evaluate nudity in images using the [NudeNet detector](https://pypi.org/project/nudenet/). The script saves images and a json files with NudeNet scores reported in the paper in the `results/` folder.

### Object Erasing

To evaluate related classes, run

`python benchmarking/object_erase.py --target <object> --baseline concept-prune --removal_mode erase --ckpt_name <path to checkpoint>`

To check interference of concept removal with unrelated classes, run

`python benchmarking/object_erase.py --target <object> --baseline concept-prune --removal_mode keep --ckpt_name <path to checkpoint>`

where `<object>` is the name of a class in ImageNette classes. The script saves images and a json files with ResNet50 accuracies reported in the paper in the `results/` folder.

## Cite us!

If you find our paper useful, please consider citing our work.

```
@article{zhang2026roots,
  title={Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models},
  author={Zhang, Ci and Ding, Zhaojun and Yang, Chence and Liu, Jun and Zhai, Xiaoming and Huang, Shaoyi and Li, Beiwen and Ma, Xiaolong and Lu, Jin and Yuan, Geng},
  journal={arXiv preprint arXiv:2603.06640},
  year={2026}
}
```
