# 扩展开发 Dino U-Net（中文）

> 作者：Yifan Gao, Haoyue Li, Feng Yuan, Xiaosong Wang*，Xin Gao*  
> 1 中国科学技术大学（合肥）  
> 2 上海创新研究院（上海）  
> 3 上海人工智能实验室（上海）  
> *通讯作者

[English](./extending.md)

本文档介绍如何在保持训练流程稳定的前提下，优雅地扩展与自定义 Dino U-Net。核心思路是：仅在 Trainer 中定制“构建网络”的方法，避免改动初始化逻辑，从而最大化复用现有管线。

- 不要重写 Trainer 的 `__init__`
- 仅重写静态方法 `build_network_architecture(...)`
- 调用训练 API 时传入 Trainer“类对象”（Class），而非字符串

## 1）替换 / 自定义网络（推荐）
在 Trainer 的 `build_network_architecture` 中返回自定义网络，整体流程保持不变：

```python
from typing import Union, List, Tuple
import torch.nn as nn
from dinounet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from dinounet_training import DinoUNet

class DinoUNetCustomTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # 方案 A：基于 DinoUNet，按需修改配置（如更换 backbone/通道数）
        config = arch_init_kwargs.get('network_config', None) or {}
        if config:
            config = {**config, 'architecture': {**config['architecture'], 'deep_supervision': enable_deep_supervision}}
            return DinoUNet.from_config(
                network_config=config,
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                dinov3_pretrained_path=None,
                dinov3_model_name='dinounet_b'
            )

        # 方案 B：直接返回你自己的 nn.Module（与现有管线兼容）
        net = DinoUNet(
            network_config=None,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            dinov3_pretrained_path=None,
            dinov3_model_name='dinounet_s'
        )
        return net
```

用法示例：
```python
from dinounet.api import plan_and_preprocess, training

plans_id, net_cfgs = plan_and_preprocess(dataset_id=4, configurations=['2d'])
result_folder, training_log = training(
    dataset_id=4,
    configuration='2d',
    trainer_class=DinoUNetCustomTrainer,  # 传类对象
    plans_identifier=plans_id,
    initial_lr=0.001,
    num_epochs=50,
    batch_size=16
)
```

## 2）更换 backbone / 新增 DINOv3 变体
在 `dinounet_training.py` 中扩展以下注册表以添加新变体：
- `DINOv3_MODEL_FACTORIES`
- `DINOv3_MODEL_INFO`
- `DINOv3_INTERACTION_INDEXES`

也可为特定变体创建便捷 Trainer 子类：
```python
class DinoUNetTrainer_xxlarge(DinoUNetCustomTrainer):
    _dinov3_model_name = 'dinounet_l'
    _dinov3_pretrained_path = 'dinounet/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
```

## 3）自定义 Adapter / 投影模块（进阶）
如需改变 ViT 特征到 U-Net 解码器的适配方式，可修改/继承：
- `DINOv3_Adapter`（特征抽取 / 交互层）
- `DINOv3EncoderAdapter`（投影到解码器通道、上采样对齐）
- `FAPM`（Feature Adaptive Projection Module）

常见改动方向：
- 调整 `FAPM` 的 rank/通道配置
- 替换上采样策略（双线性、反卷积等）
- 在跳连前加入注意力或门控模块

请确保编码器对解码器暴露以下属性：`output_channels`、`strides`、`kernel_sizes`，以便 `UNetDecoder` 正确对接。

## 4）实践建议
- 避免改动初始化：仅重写 `build_network_architecture`
- 训练入口传入“类对象”，而非字符串
- 固定 `requirements.txt` 依赖版本，并显式设置 `dinov3_pretrained_path` 有助于复现
- 若此前生成或修改过 plans，请在预处理时设置 `force_rerun=true`

---

## 致谢
- nnU-Net：通用自适应的医学图像分割框架（[项目](https://github.com/MIC-DKFZ/nnUNet)｜[数据格式](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)）
- DINOv3：自监督视觉 Transformer 及其开源实现（[项目](https://github.com/facebookresearch/dinov3)）
