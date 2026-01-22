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

