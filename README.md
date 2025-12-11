# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference ⚡  </h1>

Powered by 
<sup>1</sup> ![Peking University](https://img.shields.io/badge/Peking_University-blue?style=flat&logo=none) 
<sup>2</sup> ![Fudan University](https://img.shields.io/badge/Fudan_University-orange?style=flat&logo=none) 
<sup>3</sup> ![UC Berkeley](https://img.shields.io/badge/UC_Berkeley-green?style=flat&logo=none)
<sup>4</sup> ![The University of Sydney](https://img.shields.io/badge/The_University_of_Sydney-red?style=flat&logo=none) 
<sup>5</sup> ![Panasonic](https://img.shields.io/badge/Panasonic-purple?style=flat&logo=none) 
<sup>6</sup> ![Tsinghua University](https://img.shields.io/badge/Tsinghua_University-brown?style=flat&logo=none)

<h3 style="margin-bottom: 0; font-size: 1.5em;">
  SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
</h3>

[![Paper](https://img.shields.io/badge/V1.5%20Paper-arXiv-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2410.04417)
[![Code](https://img.shields.io/badge/V1.5%20Code-GitHub-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gumpest/SparseVLMs/tree/v1.5)

<!-- <p style="margin-top: 0; font-size: 0.6em; line-height: 1.4em;">
  <b><a href="https://gumpest.github.io/">Yuan Zhang</a></b><sup>1,3*</sup>,
  <b><a href="https://scholar.google.com/citations?user=TxeAbWkAAAAJ&hl=en&oi=ao">Chun-Kai Fan</a></b><sup>1*</sup>,
  <b><a href="https://scholar.google.com/citations?user=FH3u-hsAAAAJ&hl=en">Junpeng Ma</a></b><sup>2*</sup>,
  <b><a href="https://wzzheng.net/">Wenzhao Zheng</a></b><sup>3✉️</sup>,
  <b><a href="https://taohuang.info/">Tao Huang</a></b><sup>4</sup>,
  <b><a href="https://cfcs.pku.edu.cn/people/faculty/kuancheng/index.htm">Kuan Cheng</a></b><sup>1</sup>,<br>
  <b><a href="https://github.com/Gumpest/SparseVLMs">Denis Gudovskiy</b><sup>5</sup>,
  <b><a href="https://github.com/Gumpest/SparseVLMs">Tomoyuki Okuno</b><sup>5</sup>,
  <b><a href="https://github.com/Gumpest/SparseVLMs">Yohei Nakata</b><sup>5</sup>,
  <b><a href="http://people.eecs.berkeley.edu/~keutzer/">Kurt Keutzer</a></b><sup>3</sup>,
  <b><a href="https://idm.pku.edu.cn/info/1017/1598.htm">Shanghang Zhang</a></b><sup>1✉️</sup>
</p> -->

<h3 style="margin-bottom: 0; font-size: 1.5em;">
  SparseVLM+: Visual Token Sparsification with Improved Text-Visual Attention Pattern
</h3>

[![Paper](https://img.shields.io/badge/V2.0%20Paper-arXiv-red?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![Code](https://img.shields.io/badge/V2.0%20Code-GitHub-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gumpest/SparseVLMs)

<!-- <p style="margin-top: 0; font-size: 0.6em; line-height: 1.4em;">
  <b><a href="https://gumpest.github.io/">Yuan Zhang</a></b><sup>1*</sup>,
  <b><a href="https://scholar.google.com/citations?user=FH3u-hsAAAAJ&hl=en">Junpeng Ma</b><sup>1,2*</sup>,
  <b><a href="https://scholar.google.com/citations?user=cdAi_uIAAAAJ&hl=zh-CN">QiZhe Zhang</a></b><sup>1</sup>,
  <b><a href="https://scholar.google.com/citations?user=TxeAbWkAAAAJ&hl=en&oi=ao">Chun-Kai Fan</a></b><sup>1</sup>,<br>
  <b><a href="https://wzzheng.net/">Wenzhao Zheng</a></b><sup>3</sup>,
  <b><a href="https://cfcs.pku.edu.cn/people/faculty/kuancheng/index.htm">Kuan Cheng</a></b><sup>1</sup>,
  <b><a href="https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en">Jiwen Lu</a></b><sup>6</sup>,
  <b><a href="https://idm.pku.edu.cn/info/1017/1598.htm">Shanghang Zhang</a></b><sup>1✉️</sup>
</p> -->


## 📜 News 
🔥 **[2025/12/11]** We released new version **[SparseVLM+](https://arxiv.org/pdf/2410.04417)**! Bring a **Stronger Performance with Improved Text-Visual Attention Pattern**!

🔥 **[2025/06/04]** The sparsification code for **[VideoLLaVA](https://github.com/Gumpest/SparseVLMs/tree/video)** is now open source! Please check the `video branch`.

🔥 **[2025/05/01]** Our SparseVLM is accepted by **ICML 2025**!

🔥 **[2025/03/06]** We released **[SparseVLM v1.5](https://arxiv.org/pdf/2410.04417)**! **Higher Accuracy, Flexible Pruning Manner, and Compatibility with FlashAttention 2**!

🔥 **[2024/10/15]** We released **[SparseVLM](https://arxiv.org/pdf/2410.04417)** and its **[Project Page](https://leofan90.github.io/SparseVLMs.github.io/)**! The **[Code](https://github.com/Gumpest/SparseVLMs)** is now open-source! Please check the `v1.5 branch` for the latest version.


<p align='center'>
<img src='./assests/archi.png' alt='mask' width='700px'>
</p>

## ✒️ Contents
- [News](#news)
- [Contents](#contents)
- [Overview](#overview)
- [Preparation](#preparation)
- [Usage](#usage)
- [SparseVLM+](#sparsevlm+)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

## 👀 Overview

In vision-language models (VLMs), visual tokens usually consume a significant amount of computational overhead, despite their sparser information density compared to text tokens. To address this, existing methods extract more compact image representations by modifying the image encoder or projector. While some recent works further sparsify vision tokens during the decoding, they still ignore the guidance from the language tokens, which **contradicts the multimodality paradigm**. We argue that **visual tokens should be sparsified adaptively based on the question prompt**, as the model might focus on different parts (e.g., foreground or background) when dealing with various questions, as shown in Figure below. Unlike previous methods with text-agnostic visual sparsification (c) e.g., recent FastV, our SparseVLM (b) is guided by question prompts to select relevant visual patches.

<div align=center>
<img width="600" alt="image" src="./assests/moti.png">
</div>

## 👨‍💻 Preparation

1. Clone this repository and navigate to SparseVLMs folder
```bash
git clone https://github.com/Gumpest/SparseVLMs.git
cd SparseVLMs
```

2. Install necessary package
```Shell
conda create -n SparseVLMs python=3.10 -y
conda activate SparseVLMs
pip install -e .
pip install transformers==4.37.0
pip install flash_attn==2.3.3
```

3. Download Multimodal Benchmark

Please follow the detailed instruction in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## 🎯 Basic Usage
Specifically, setting `RETAIN_TOKN` in the environment variables indicates the number of tokens to be retained after the SparseVLM algorithm. It supports four numbers of tokens, including **192, 128, 96, and 64**. If a specific number of tokens is required, please make modifications in `./llava/model/language_model/score.py`

1. Example for evaluating MME results (retain 192 tokens):
```Shell
RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh
```

2. Example for evaluating TextVQA results (retain 128 tokens):
```Shell
RETAIN_TOKN=128 bash scripts/v1_5/eval/textvqa.sh
```

3. Example for evaluating ScienceQA results (retain 96 tokens):
```Shell
RETAIN_TOKN=96 bash scripts/v1_5/eval/sqa.sh
```

4. Example for evaluating MMBench results (default 64 tokens):
```Shell
RETAIN_TOKN=64 bash scripts/v1_5/eval/mmbench.sh
```

## 🛠️ One-Click Enable SparseVLM+ (V2.0 Mode)
You can boost the performance of SparseVLM by enabling the V2.0 mode, which can be seamlessly enabled via an environment variable without modifying the code.

1. Example for evaluating MME results (retain 192 tokens):
```Shell
USE_VERSION=2_0 RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh
```

2. Example for evaluating TextVQA results (retain 128 tokens):
```Shell
USE_VERSION=2_0 RETAIN_TOKN=128 bash scripts/v1_5/eval/textvqa.sh
```

3. Example for evaluating MMBench results (retain 96 tokens):
```Shell
USE_VERSION=2_0 RETAIN_TOKN=96 bash scripts/v1_5/eval/mmbench.sh
```

4. Example for evaluating GQA results (retain 64 tokens):
```Shell
USE_VERSION=2_0 RETAIN_TOKN=64 bash scripts/v1_5/eval/gqa.sh
```

## License
This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use SparseVLM in your research, please cite our work by using the following BibTeX entry:
```bibtex
@inproceedings{zhang2024sparsevlm,
  title={SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference},
  author={Zhang, Yuan and Fan, Chun-Kai and Ma, Junpeng and Zheng, Wenzhao and Huang, Tao and Cheng, Kuan and Gudovskiy, Denis and Okuno, Tomoyuki and Nakata, Yohei and Keutzer, Kurt and others},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

```

```bibtex
@inproceedings{zhang2025sparsevlmplus,
  title={SparseVLM+: Visual Token Sparsification with Improved Text-Visual Attention Pattern},
  author={Zhang, Yuan and Ma, Junpeng and Zhang, Qizhe and Fan, Chun-Kai and Zheng, Wenzhao and Cheng, Kuan and Lu, Jiwen and Zhang, Shanghang},
  year={2025}
}

```
## Acknowledgment

We extend our gratitude to the open-source efforts of [TCFormer](https://github.com/zengwang430521/TCFormer), [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGemini](https://github.com/dvlab-research/MGM) and [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA).
