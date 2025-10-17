# 🧩 Based-on-point-cloud-partitions

> **A strategy for dividing data after sparse reconstruction based on the number of point clouds, designed for 3DGS reconstruction.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

This project provides a strategy to **partition point cloud data** after sparse reconstruction.  
It is primarily intended for use in **3D Gaussian Splatting (3DGS)** reconstruction pipelines, enabling efficient division and management of large-scale 3D data.
<p align="center">
  <img src="./assets/render01.png" alt="Partition Result Example" width="600">
</p>


<p align="center">
  <img src="./assets/render02.png" alt="Partition Result Example" width="600">
</p>

### 🎬 训练结果演示视频
![](./assets/demo.gif)
[点击这里下载视频](https://github.com/92401/Based-on-point-cloud-partitions/tree/main/assets/demo.mp4)

---

## 📦 Installation

Clone this repository (including submodules):

```bash
# SSH
git clone --recursive https://github.com/1799967694/Based-on-point-cloud-partitions.git
```

---

## ⚙️ Setup

### 🪟 For Windows users

Before installing dependencies, set the following environment variable:

```bash
SET DISTUTILS_USE_SDK=1
```

### 🧰 Create and activate the conda environment

```bash
conda env create --file environment.yml
conda activate ptgs
```

### ▶️ Run the script

```bash
cd Based-on-point-cloud-partitions/scene/ptgs
$env:PYTHONPATH="your_path/Based-on-point-cloud-partitions"
# Set the code as an environment variable for easy access
python shen_partition_utils.py your_sfm_path
```

---

## 📝 Notes

Here are some important tips to keep in mind when running the code:

### 1. Partition Strategy
- By default, the partitioning is performed **along the XY-plane**.
- The **Manhattan rotation matrix** is **not applied** by default.  
  If you want to enable it, simply **uncomment the `man_trans` section** in `shen_partition_utils.py`.
- For details on determining the **Manhattan parameters**, please refer to the following repository:  
  🔗 [VastGaussian-refactor](https://github.com/1799967694/VastGaussian-refactor)

---

### 2. Threshold Parameter (`threshold_value`)
The parameter `threshold_value = 500000` is related to the **available GPU memory** during training.  
You can adjust this value according to your GPU’s VRAM capacity.

| GPU Memory | Recommended `threshold_value` |
|-------------|-------------------------------|
| 24 GB       | 500,000                       |
| 12 GB       | 200,000                       |
| 8 GB        | 100,000                       |

> 💡 **Tip:** If you encounter memory overflow issues, try lowering `threshold_value` accordingly.

---

## 📸 Example Output

Before running the script, please ensure that your input directory follows the structure below:

```
your_sfm/
├── images
├── sparse
```

After successful execution of the partitioning script, the following folder will be generated:

```
your_sfm/model/
└── split_result/
    └── visible/        # Stores the visualization results for each partition
```

Each subfolder under `visible/` represents **one independent 3DGS input dataset**,  
and the corresponding `.pkl` files record the **partitioning information** for each region.

Batch Training and Merging

You can batch train all partitioned sub-blocks using the following command:

```bash
python auto_train.py --base_path E:\model\partition_point_cloud\visible

```
---

### 🖼️ Visualization Example

Below are sample visualizations of the partitioning results:

<p align="center">
  <img src="./assets/partition_result_1.png" alt="Partition Result Example" width="600">
</p>

> **Figure:** Visualization of the generated partitions under `model/split_result/visible/`.  
> Each folder corresponds to a distinct region of the point cloud that can be processed independently.

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---


