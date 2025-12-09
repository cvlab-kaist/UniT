<div align="center">
<h1>
UniT: Unified Diffusion Transformer for High-fidelity Text-Aware Image Restoration</h1>


[**Jin Hyeon Kim**](https://github.com/jinlovespho)<sup>1</sup>,
**Paul Hyunbin Cho**<sup>1</sup>, 
**Claire Kim**<sup>1</sup>,

[**Jaewon Min**](https://github.com/Min-Jaewon/)<sup>1</sup>, 
[**Jaeeun Lee**](https://github.com/babywhale03)<sup>2</sup>,
**Jihye Park**<sup>2</sup>, **Yeji Choi**<sup>1</sup>, [**Seungryong Kim**](https://scholar.google.com/citations?hl=zh-CN&user=cIK1hS8AAAAJ)<sup>1</sup>

<sup>1</sup> KAIST&nbsp;AI 

<sup>2</sup> Samsung&nbsp;Electronics

<a href=""><img src="https://img.shields.io/badge/arXiv-2512"></a>
        <a href="https://cvlab-kaist.github.io/UniT/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>
</div>


# 🔈 News 
- [ ] 📄 Arxiv paper coming soon.
- [ ] 🔥Training code coming soon.
- [x] ⚔️ **2025.12.10** — Demo inference code released!
- [x] 🚀 **2025.12.09** — Official launch of the repository and project page!




#  👟 Installation Walkthrough


**1. Conda Environment**

```bash
conda create -n unit python=3.9 -y
conda activate unit
```

**2. Library Installation**
- Download the following libraries in the order listed below.
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy==1.26.3 --no-deps
pip install pyiqa==0.1.14.1 --no-deps 
cd transformers
pip install -e .
cd ../detectron2 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
cd ../testr 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
pip install cloudpickle --no-deps
```

**3. Model Pretrained Weights**

Download the respective model weights by running the respective bash files below.

- **Stable Diffusion3 Weights (vae, noise scheduler, tokenizer, etc)**
```bash
# Authenticate with your HF access token
huggingface-cli login   
# Then run the bash file
bash download_bash/download_sd3.sh 
```

- **Text Spotting Module (TSM) Weights**
```bash
# Required for training only
bash download_bash/download_testr.sh 
```

- **Diffusion Transformer (DiT) Weights**
```bash
# Required for training only
bash download_bash/download_testr.sh 
```





# 🚀 Inference Demo
### Demo Script 
Download the released UniT weight from [google drive](https://drive.google.com/drive/folders/1JJOdEgM-rdu9GYGifG4kFxKF66JMeagR?usp=drive_link), and set the corresponding path in the demo [configuration file](run_configs/val/val_unit_demo.yaml). Then, run the script below to perform text-aware image restoration on low-quality image samples. The results will be saved in result_val/ by default.
```bash
bash run_scripts/val/run_val_unit_demo.sh
```


### Demo Result
Running the demo inference script will generate the following text restoration results. The visualized images are shown in the order: **Low-Quality image / Restored High-Quality image / Ground Truth image.** Checkout our [Project Page](https://cvlab-kaist.github.io/UniT/) for full text restoration qualitative results!
<p align="center">
  <img src="assets/demo_result/demo1.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo2.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo3.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo4.png" width="800">
</p>

<!-- 
# 🔥 Training Recipe  

### 1. Data Preparation
- First, download the SA-Text training data from this [repo](https://github.com/cvlab-kaist/TAIR).

### 2. Model Weight Preparation
- Second, prepare the DiT and TSM model weights as described in section **👟Installation Walkthrough - 3.Model Pretrained Weights**.
- The model weights will be automatically downloaded into the **preset/models/** directory.



### 3. Training Script
Modify both the training bash script and the YAML configuration file before starting training.

- Stage 1 training bash file: [run_train_unit_stage1.sh](run_scripts/train/run_train_unit_stage1.sh)
- Stage 1 yaml config file: [train_unit_stage1.yaml](run_configs/train/train_unit_stage1.yaml)
```bash
# stage1 training 
bash run_scripts/train/JIHYE_train_stage1_dit4sr.sh
```


- Stage 2 training bash file: [run_train_unit_stage2.sh](run_scripts/train/run_train_unit_stage2.sh)
- Stage 2 yaml config file: [train_unit_stage2.yaml](run_configs/train/train_unit_stage2.yaml)
```bash
# stage2 training 
bash run_scripts/train/JIHYE_train_stage2_testr.sh
``` -->


## Citation

```

```

## Acknowledgement
We thank the authors of [DiT4SR](https://github.com/facebookresearch/vggt), [TAIR](https://github.com/cvlab-kaist/TAIR), and [TESTR](https://github.com/mlpc-ucsd/TESTR) for their excellent work and code, which served as the foundation for this project.