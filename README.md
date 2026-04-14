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
