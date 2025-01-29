# Token-based Decision Criteria Are Suboptimal in In-context Learning

<p align="center">
  <a href="https://www.hakaze-c.com/">Hakaze Cho</a>, et al.
  <br>
  <br>
  <a href="https://github.com/hc495/ICL_Circuit/blob/master/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2Fhc495%2FICL_Circuit%2Fblob%2Fmaster%2FLICENSE"></a>
  <a href="https://openreview.net/forum?id=xizpnYNvQq"><img src="https://img.shields.io/badge/NAACL_2025-Accepted_(Main)-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DxizpnYNvQq"></a>
  <a href="https://arxiv.org/abs/2406.16535"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2406.16535-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2410.04468"></a>
</p>

**This repo contains the official code for the following paper published at NAACL 2025 Main conference:**

> Hakaze Cho, et al. **"Token-based Decision Criteria Are Suboptimal in In-context Learning."** *The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL (NAACL): Main conference*, 2025.

Implemented by [Hakaze Cho](https://www.hakaze-c.com/), the primary contributor of the paper.

Some reloaded version of Hidden Calibration can be found in [StaICC](https://github.com/hc495/StaICC/blob/dc9100bb0a2738847bee5c671022377f7a7cdd46/prefabricate_inference/standard_calibration.py#L137) or [ICL_Circuit](https://github.com/hc495/ICL_Circuit/blob/master/Experiments/Exp2_Centroid_Classifier.ipynb).

## Overview

### Abstract

*In-Context Learning (ICL) typically utilizes classification criteria from output probabilities of manually selected label tokens. However, we argue that such token-based classification criteria lead to suboptimal decision boundaries, despite delicate calibrations through translation and constrained rotation applied. To address this problem, we propose Hidden Calibration, which renounces token probabilities and uses the nearest centroid classifier on the LM's last hidden states. In detail, we assign the label of the nearest centroid previously estimated from a calibration set to the test sample as the predicted label. Our experiments on 6 models and 10 classification datasets indicate that Hidden Calibration consistently outperforms current token-based baselines by about 20%~50%, achieving a strong state-of-the-art in ICL. Our further analysis demonstrates that Hidden Calibration finds better classification criteria with less inter-class overlap, and LMs provide linearly separable intra-class clusters with the help of demonstrations, which supports Hidden Calibration and gives new insights into the principle of ICL.*

### Summary figure

![Summary figure](https://s2.loli.net/2025/01/29/JSEDulIhvVgenL5.png)

In an ICL diagram, **A.** The prompt of ICL consists of a combination of demonstrations and a query. LMs encode the prompt into the last hidden state $h$, then **B.** Previous works use the un-embedding vectors of the label tokens ($E^U_+$ and $E^U_-$) to decode the $h$ to prediction $\hat{y}$, then calibrations are used to adjust the predicted logits. **C.** Our work uses the calibration dataset to calculate centroids ($\bar{h}_+$ and $\bar{h}_-$) to decode the $h$.

## Set Up

### 0. Requirement

1. A GPU with more than 22GB VRAM and CUDA (Ver. `12.4` recommended) are strongly required to run all the experiments.
2. Network connection to `huggingface` is needed to download the pre-trained model. And a `huggingface` user token with access to the [`Llama2`](https://huggingface.co/meta-llama/Llama-2-7b) model is recommended to run a part of the experiments.
3. `Anaconda` or `Miniconda` is needed.

### 1. Clone the repository

```bash
git clone https://github.com/hc495/Hidden_Calibration.git
```

### 2. Environment Installation

```bash
conda env create -f environment.yaml
conda activate hidden_calibration
```

### 3. Make Sure Your Working Directory is the Root Directory of the Project

You need to ensure that your working directory is set to the root directory of the project, i.e., the same directory as `README.md`, even if you open a Jupyter notebook from the `Experiments` folder.

We provide a default `os.chdir()` method in every notebook, you should use it to move the working directory to the root directory.