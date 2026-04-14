# Roots Beneath the Cut

Code for the paper - **Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models**

[![arXiv](https://img.shields.io/badge/arXiv-2603.06640-b31b1b.svg)](https://arxiv.org/abs/2603.06640)

## Introduction

Pruning-based unlearning has recently emerged as a fast, training-free, and data-independent approach to remove undesired concepts from diffusion models. It promises high efficiency and robustness, offering an attractive alternative to traditional fine-tuning or editing-based unlearning. However, in this paper we uncover a hidden danger behind this promising paradigm. We find that the locations of pruned weights, typically set to zero during unlearning, can act as side-channel signals that leak critical information about the erased concepts. To verify this vulnerability, we design a novel attack framework capable of reviving erased concepts from pruned diffusion models in a fully data-free and training-free manner. Our experiments confirm that pruning-based unlearning is not inherently secure, as erased concepts can be effectively revived without any additional data or retraining. Extensive experiments on diffusion-based unlearning based on concept related weights lead to the conclusion: once the critical concept-related weights in diffusion models are identified, our method can effectively recover the original concept regardless of how the weights are manipulated. Finally, we explore potential defense strategies and advocate safer pruning mechanisms that conceal pruning locations while preserving unlearning effectiveness, providing practical insights for designing more secure pruning-based unlearning frameworks.

<p align="center">
<img src="Image/Picture_hacker.png" width="50%">
</p>

## Experiments

### Environment Setup

Create Environment from the `environment.yml` file.

`cd env`

`conda env create -f environment.yml`

`conda activate concept_revival`

### Obtain the Unlearned Model

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

<p align="center">
<img src="Image/unlearn.png" width="50%">
</p>
