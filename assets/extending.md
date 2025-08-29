# Extending Dino U-Net

> Authors: Yifan Gao, Haoyue Li, Feng Yuan, Xiaosong Wang*, and Xin Gao*  
> 1 University of Science and Technology of China, Hefei, China  
> 2 Shanghai Innovation Institute, Shanghai, China  
> 3 Shanghai Artificial Intelligence Laboratory, Shanghai, China  
> *Corresponding author

This guide explains how to extend and customize Dino U-Net for your research. The design follows a clean API similar to our simplified nnU-Net wrapper: customize the trainer's network-building method without changing initialization logic.

- Do NOT override `__init__` in trainer classes.
- Only override the static method `build_network_architecture(...)`.
- Pass the trainer CLASS (not string) to the training API.

### 1) Replace/Customize the Network (Recommended Entry Point)

You can keep the existing pipeline and return a customized network from the trainer's `build_network_architecture`.

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
        # Option A: Use DinoUNet with modified config (e.g., different backbone or channels)
        config = arch_init_kwargs.get('network_config', None) or {}
        if config:
            # Ensure deep supervision flag follows the trainer
            config = {**config, 'architecture': {**config['architecture'], 'deep_supervision': enable_deep_supervision}}
            return DinoUNet.from_config(
                network_config=config,
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                dinov3_pretrained_path=None,         # or a custom path
                dinov3_model_name='dinounet_b'       # change to 'dinounet_s'/'dinounet_l'/'dinounet_7b'
            )

        # Option B: Return your own nn.Module compatible with the pipeline
        class SimpleHead(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, 1)
            def forward(self, x):
                return self.conv(x)

        # Example: wrap DinoUNet and add a custom prediction head if needed
        net = DinoUNet(
            network_config=None,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            dinov3_pretrained_path=None,
            dinov3_model_name='dinounet_s'
        )
        return net
```

Usage:
```python
from dinounet.api import plan_and_preprocess, training

plans_id, net_cfgs = plan_and_preprocess(dataset_id=4, configurations=['2d'])
result_folder, training_log = training(
    dataset_id=4,
    configuration='2d',
    trainer_class=DinoUNetCustomTrainer,   # pass CLASS
    plans_identifier=plans_id,
    initial_lr=0.001,
    num_epochs=50,
    batch_size=16
)
```

### 2) Swap Backbone or Add New DINOv3 Variant

Dino U-Net maps model keys to backbones via an internal registry (see `DINOv3_MODEL_FACTORIES`, `DINOv3_MODEL_INFO`, and `DINOv3_INTERACTION_INDEXES` in `dinounet_training.py`). To add a new variant:

- Register a new key in `DINOv3_MODEL_FACTORIES` with a factory function.
- Add its metadata to `DINOv3_MODEL_INFO`.
- Define appropriate `interaction_indexes` in `DINOv3_INTERACTION_INDEXES`.
- Optionally create a dedicated trainer subclass (see below) for convenience.

```python
class DinoUNetTrainer_xxlarge(DinoUNetCustomTrainer):
    """Example: a convenience trainer bound to a specific backbone/checkpoint."""
    _dinov3_model_name = 'dinounet_l'  # or your new key
    _dinov3_pretrained_path = 'dinounet/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
```

Then use it in the same way by passing the class to `training(...)`.

### 3) Customize Adapter/Projection (Advanced)

If you need to change how ViT features are adapted to the U-Net decoder, modify or subclass:
- `DINOv3_Adapter` (feature extraction / interaction layers)
- `DINOv3EncoderAdapter` (projection to decoder feature channels, upsampling)
- `FAPM` (Feature Adaptive Projection Module)

Typical customizations include:
- Changing ranks/channels in `FAPM`
- Replacing bilinear or transpose upsampling
- Adding attention or gating blocks before skip connections

When you introduce new modules, ensure the encoder exposes:
- `output_channels`, `strides`, `kernel_sizes`
so that `UNetDecoder` can connect correctly.

### 4) Practical Tips

- Keep trainer initialization untouched; only override `build_network_architecture`.
- Pass the trainer CLASS to the training API.
- For reproducible experiments, use explicit `dinov3_pretrained_path` and pin versions in `requirements.txt`.
- If you modified preprocessing or plans before, remember to set `force_rerun=true` during preprocessing.

---

## Acknowledgements

We thank the following open-source projects that made this work possible:
- nnU-Net: Universal self-adapting framework for U-Net-based medical image segmentation ([repo](https://github.com/MIC-DKFZ/nnUNet), [dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)).
- DINOv3: Self-supervised Vision Transformers used as our encoder backbone ([repo](https://github.com/facebookresearch/dinov3)).


