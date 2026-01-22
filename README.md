# SAR-IceFM: Foundation Models for SAR-Based Arctic Sea Ice Analysis

## Overview

**SAR-IceFM** is a research framework for the development and evaluation of **foundation models for Arctic sea ice analysis using Synthetic Aperture Radar (SAR)**.  
The repository integrates **self-supervised pre-training** and **multi-task downstream segmentation** pipelines designed to address key challenges in operational sea-ice mapping, including limited labeled data, coarse polygon-based supervision, and complex SAR backscatter behavior.

The framework is implemented within the **OpenMMLab** ecosystem and extends existing toolboxes to support **remote sensing–specific data handling**, **multi-task learning**, and **large-scale experimentation** on Arctic SAR datasets.

---

## Scientific Context and Motivation

The increasing availability of satellite observations, coupled with the scarcity of high-quality labeled data, motivates the development of **foundation models** for automated sea-ice analysis. SAR is the primary data source for operational ice charting; however, its use poses challenges due to heterogeneous backscatter responses and the mismatch between pixel-level predictions and polygon-based ice-chart annotations.

Using the **AI4Arctic** dataset, SAR-IceFM investigates the feasibility and limitations of **Masked Autoencoder (MAE)** pre-training for SAR imagery, followed by fine-tuning on three downstream tasks:

- **Sea Ice Concentration (SIC)**
- **Stage of Development (SOD)**
- **Floe Size (FLOE)**

Across a range of data regimes and model capacities, MAE pre-training consistently improves downstream performance relative to training from scratch. The largest gains reach **up to 6.6% in a combined evaluation metric**, particularly for larger backbones and when sufficient supervised data are available for fine-tuning.

The framework further enables systematic analyses of:
- Single-task versus multi-task fine-tuning  
- Resolution transferability under single- and multi-scale training  
- Sensitivity to MAE mask ratio  
- Limitations of shared encoders under coarse supervision  

Overall, the results indicate that while self-supervised pre-training is beneficial, downstream performance remains strongly influenced by **annotation quality**, **task heterogeneity**, and **resolution handling**.

---

## Repository Structure

This public repository aggregates two complementary components, each documented independently:


---

## Components

### Self-Supervised Pre-Training (`pre-training/`)

The **pre-training** module extends  
[mmselfsup](https://github.com/open-mmlab/mmselfsup)  
to support **SAR-based self-supervised representation learning** for Arctic sea-ice imagery. The implementation includes MAE-based pre-training pipelines and visualization tools for analyzing reconstructions on the AI4Arctic dataset.

**Documentation:**  
→ [`pre-training/README.md`](./pre-training/README.md)

Typical use cases include:
- Learning SAR-specific representations without labeled data  
- Studying the effect of mask ratio, spatial resolution, and model capacity  
- Generating pretrained encoders for downstream segmentation tasks  

---

### Multi-Task Downstream Segmentation (`downstream-seg/`)

The **downstream segmentation** module extends  
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
to support **multi-task semantic segmentation** of Arctic sea ice. A shared encoder is coupled with **task-specific decoders** to jointly predict:

- Sea Ice Concentration (regression)
- Stage of Development (classification)
- Floe Size (classification)

The module includes:
- Custom datasets and patch-based pipelines for AI4Arctic  
- Multi-task segmentors and decode heads  
- Regression-aware losses and multi-task evaluation metrics  
- Modified training, logging, and visualization hooks tailored to Arctic SAR data  

**Documentation:**  
→ [`downstream-seg/README.md`](./downstream-seg/README.md)

---

## Typical Workflow

A standard experimental workflow within SAR-IceFM is:

1. **Self-supervised pre-training** of a SAR encoder using the code in `pre-training/`
2. **Fine-tuning** of the pretrained encoder on SIC, SOD, and FLOE using `downstream-seg/`
3. **Evaluation** across different data regimes, resolutions, and task configurations

Each module provides detailed instructions, environment setup scripts, and example training loops, including configurations designed for **HPC environments** (e.g., Compute Canada).

---

## Pretrained Models and Experimental Artifacts

Pretrained weights and experimental checkpoints are **not stored in the git history**.  
They are distributed via **GitHub Releases** associated with this repository.

→ [v1.0.0](https://github.com/jnoat92/SAR-IceFM/releases/tag/v1.0.0)

Each release includes:
- Pretrained model weights  
- Corresponding configuration files  
- Checksums for reproducibility  

---

## Citation

If you use SAR-IceFM in your research, please cite the corresponding publication describing the MAE-based foundation model experiments on the AI4Arctic dataset.

*(BibTeX entry to be added upon publication or preprint release.)*

---




