from dinounet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch._dynamo import OptimizedModule


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        plans_identifier: str = 'nnUNetPlans',
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
