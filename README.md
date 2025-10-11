# 🧩 Based-on-point-cloud-partitions

> **A strategy for dividing data after sparse reconstruction based on the number of point clouds, designed for 3DGS reconstruction.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

This project provides a strategy to **partition point cloud data** after sparse reconstruction.  
It is primarily intended for use in **3D Gaussian Splatting (3DGS)** reconstruction pipelines, enabling efficient division and management of large-scale 3D data.

---

## 📦 Installation

Clone this repository (including submodules):

```bash
# SSH
git clone --recursive https://github.com/1799967694/Based-on-point-cloud-partitions.git

## ⚙️ Setup

### 🪟 For Windows users
Before installing dependencies, set the following environment variable:

```bash
SET DISTUTILS_USE_SDK=1

conda env create --file environment.yml
conda activate ptgs

cd Based-on-point-cloud-partitions/scene/ptgs
python shen_partition_utils.py your_sfm_path
