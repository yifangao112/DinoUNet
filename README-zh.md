# Dino U-Net（中文）

[English](./README.md)

这是 Dino U-Net 的官方实现：一种在 U-Net 架构中集成 DINOv3 预训练模型的医学图像分割框架。通过利用 DINOv3 的高保真密集特征，Dino U-Net 在多种医学影像任务上取得优异表现。

> 论文：**[Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation](https://arxiv.org/pdf/2508.20909)**
> <br>作者：Yifan Gao, Haoyue Li, Feng Yuan, Xiaosong Wang*, Xin Gao*<br>
> 1 中国科学技术大学<br> 2 上海创智学院<br> 3 上海人工智能实验室<br>

## 特性
- **基础模型编码器**：采用 DINOv3 作为高保真特征提取器
- **多规模模型可选**：ViT-S/B/L/7B，性能与算力灵活权衡
- **集成 nnU-Net**：统一的数据预处理、训练与评估流程
- **高性能**：将通用视觉预训练能力迁移到医学分割

## 依赖
- Python 3.8+
- PyTorch 1.10+
- CUDA GPU

## 安装
1. 克隆代码库
```bash
git clone https://github.com/yifangao112/DinoUNet.git
cd dino-unet
```
2. 创建并激活虚拟环境（示例）
```bash
conda create -n dinounet python=3.10 -y
conda activate dinounet
```
3. 安装 PyTorch（按你的 CUDA 版本选择官方命令）
4. 安装项目依赖
```bash
pip install -r requirements.txt
```
5. 安装 MultiScaleDeformableAttention 模块
```bash
cd dinounet/dinov3/eval/segmentation/models/utils/ops
pip install .
```
6. 下载 DINOv3 预训练权重至 `dinounet/checkpoints/`

## 数据集准备
本项目采用经修改的 **nnU-Net** 框架进行数据处理。请按 nnU-Net 的数据格式组织数据（参考官方文档）：
[数据格式指南](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)。

环境变量设置（Linux/macOS 示例，追加到 `~/.bashrc` 或 `~/.zshrc`）：
```bash
export nnUNet_raw="/path/to/your/raw_data"
export nnUNet_preprocessed="/path/to/your/preprocessed_data"
export nnUNet_results="/path/to/your/model_results"
```
更多平台与持久化设置请参考：
[设置环境变量](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md)。

## 训练
示例命令：
```bash
python dinounet_training.py --gpuid 0 --model dinounet_s --datasetid 9 --epoch 200
```
脚本将自动：
1. 预处理数据
2. 配置网络结构
3. 训练模型
4. 将结果与日志保存到 `nnUNet_results`

注意：若此前已生成过 nnU-Net 计划（plans），请在预处理阶段将 `force_rerun=true` 以避免使用过期缓存。

## 评估
训练结束后自动进行评估，并在控制台与结果目录输出指标（如 Dice、HD95）。

## 扩展开发
- 英文扩展指南：`assets/extending.md`
- 中文扩展指南：`assets/extending_zh.md`

## 引用
如果本项目对您的研究有帮助，请引用：
```bibtex
@article{gao2025dino,
  title={Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation},
  author={Gao, Yifan and Li, Haoyue and Yuan, Feng and Wang, Xiaosong and Gao, Xin},
  journal={arXiv preprint arXiv:2508.20909},
  year={2025},
  url={https://arxiv.org/pdf/2508.20909}
}
```

## 致谢
- nnU-Net：通用、可自适应的医学图像分割框架（[项目主页](https://github.com/MIC-DKFZ/nnUNet)｜[数据格式](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)）
- DINOv3：自监督视觉 Transformer 及其实现（[项目仓库](https://github.com/facebookresearch/dinov3)）
