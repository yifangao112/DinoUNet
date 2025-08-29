from typing import Union, List, Optional, Type, Tuple
import multiprocessing

import torch
from dinounet.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from dinounet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dinounet.run.run_training import run_training, get_trainer_from_args
from dinounet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from dinounet.evaluation.evaluate_predictions import compute_metrics_on_folder2
from dinounet.paths import nnUNet_preprocessed
from dinounet.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import *


def _extract_training_log(logger):
    """
    从训练器的logger中提取训练日志

    Args:
        logger: nnUNetLogger实例

    Returns:
        dict: 包含训练日志的字典
    """
    if logger is None or not hasattr(logger, 'my_fantastic_logging'):
        return {
            'epochs': [],
            'train_losses': [],
            'val_losses': []
        }

    log_data = logger.my_fantastic_logging
    num_epochs = len(log_data.get('train_losses', []))

    return {
        'epochs': list(range(num_epochs)),
        'train_losses': log_data.get('train_losses', []),
        'val_losses': log_data.get('val_losses', [])
    }


def _extract_network_configurations(dataset_id: Union[int, List[int]],
                                   plans_identifier: str,
                                   configurations: List[str]) -> dict:
    """
    从plans文件中提取详细的网络配置信息

    Args:
        dataset_id: 数据集ID或ID列表
        plans_identifier: plans文件标识符
        configurations: 要提取配置的列表

    Returns:
        dict: 包含每个配置的详细网络配置信息
    """
    from dinounet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from batchgenerators.utilities.file_and_folder_operations import join, load_json

    # 确保dataset_id是单个值
    if isinstance(dataset_id, list):
        dataset_id = dataset_id[0]  # 取第一个数据集ID

    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    plans_file = join(nnUNet_preprocessed, dataset_name, f"{plans_identifier}.json")

    if not isfile(plans_file):
        print(f"Warning: Plans file not found at {plans_file}")
        return {}

    try:
        plans = load_json(plans_file)
        network_configurations = {}

        for config_name in configurations:
            if config_name in plans['configurations']:
                config = plans['configurations'][config_name]

                # 提取架构信息
                arch_info = config.get('architecture', {})
                arch_kwargs = arch_info.get('arch_kwargs', {})

                # 提取数据配置信息
                data_config = {
                    'batch_size': config.get('batch_size', None),
                    'patch_size': config.get('patch_size', []),
                    'spacing': config.get('spacing', []),
                    'median_image_size_in_voxels': config.get('median_image_size_in_voxels', [])
                }

                # 构建网络配置信息
                network_config = {
                    'architecture': {
                        'network_class_name': arch_info.get('network_class_name', ''),
                        'n_stages': arch_kwargs.get('n_stages', 0),
                        'features_per_stage': arch_kwargs.get('features_per_stage', []),
                        'kernel_sizes': arch_kwargs.get('kernel_sizes', []),
                        'strides': arch_kwargs.get('strides', []),
                        'n_conv_per_stage': arch_kwargs.get('n_conv_per_stage', []),
                        'n_conv_per_stage_decoder': arch_kwargs.get('n_conv_per_stage_decoder', []),
                        'conv_op': arch_kwargs.get('conv_op', ''),
                        'norm_op': arch_kwargs.get('norm_op', ''),
                        'nonlin': arch_kwargs.get('nonlin', ''),
                        'conv_bias': arch_kwargs.get('conv_bias', True),
                        'dropout_op': arch_kwargs.get('dropout_op', None),
                        'norm_op_kwargs': arch_kwargs.get('norm_op_kwargs', {}),
                        'nonlin_kwargs': arch_kwargs.get('nonlin_kwargs', {}),
                        'dropout_op_kwargs': arch_kwargs.get('dropout_op_kwargs', {})
                    },
                    'data_config': data_config
                }

                network_configurations[config_name] = network_config
            else:
                print(f"Warning: Configuration '{config_name}' not found in plans file")

        return network_configurations

    except Exception as e:
        print(f"Error loading network configurations: {e}")
        return {}


def _load_training_log_from_folder(output_folder):
    """
    从输出文件夹中加载训练日志

    Args:
        output_folder: 训练输出文件夹路径

    Returns:
        dict: 包含训练日志的字典
    """
    # 尝试从checkpoint中加载logger信息
    checkpoint_files = ['checkpoint_final.pth', 'checkpoint_best.pth', 'checkpoint_latest.pth']

    for checkpoint_file in checkpoint_files:
        checkpoint_path = join(output_folder, checkpoint_file)
        if isfile(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'logger' in checkpoint:
                    logger_data = checkpoint['logger']
                    num_epochs = len(logger_data.get('train_losses', []))
                    return {
                        'epochs': list(range(num_epochs)),
                        'train_losses': logger_data.get('train_losses', []),
                        'val_losses': logger_data.get('val_losses', [])
                    }
            except Exception as e:
                print(f"Warning: Could not load training log from {checkpoint_file}: {e}")
                continue

    # 如果无法从checkpoint加载，返回空日志
    print("Warning: Could not load training log from any checkpoint file")
    return {
        'epochs': [],
        'train_losses': [],
        'val_losses': []
    }


def _run_extract_fingerprints(dataset_id, fingerprint_extractor_class, num_processes,
                             verify_dataset_integrity, clean, verbose):
    """
    在单独的函数中运行extract_fingerprints，以便可以在if __name__ == '__main__'保护下运行
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Extract fingerprints
    print("Running fingerprint extraction...")
    extract_fingerprints(dataset_id, fingerprint_extractor_class, num_processes, 
                        verify_dataset_integrity, clean, verbose)


def _run_plan_experiments(dataset_id, experiment_planner_class, gpu_memory_target, 
                         preprocessor_name, overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages):
    """
    在单独的函数中运行plan_experiments，以便可以在if __name__ == '__main__'保护下运行
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Plan experiments
    print("Running experiment planning...")
    return plan_experiments(dataset_id, experiment_planner_class, 
                           gpu_memory_target, preprocessor_name,
                           overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages)


def _run_preprocess(dataset_id, plans_identifier, configurations, num_processes, verbose):
    """
    在单独的函数中运行preprocess，以便可以在if __name__ == '__main__'保护下运行
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Run preprocessing
    print("Running preprocessing...")
    preprocess(dataset_id, plans_identifier, configurations, num_processes, verbose)


def _check_preprocessing_completed(dataset_id: Union[int, List[int]],
                                 plans_identifier: str,
                                 configurations: List[str]) -> bool:
    """
    检查指定数据集和配置的预处理是否已经完成

    Args:
        dataset_id: 数据集ID或ID列表
        plans_identifier: plans文件标识符
        configurations: 要检查的配置列表

    Returns:
        bool: 如果所有配置的预处理都已完成则返回True，否则返回False
    """
    from dinounet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    # 确保dataset_id是列表
    if isinstance(dataset_id, int):
        dataset_ids = [dataset_id]
    else:
        dataset_ids = dataset_id

    for did in dataset_ids:
        dataset_name = maybe_convert_to_dataset_name(did)
        preprocessed_folder = join(nnUNet_preprocessed, dataset_name)

        # 检查fingerprint文件是否存在
        fingerprint_file = join(preprocessed_folder, 'dataset_fingerprint.json')
        if not isfile(fingerprint_file):
            return False

        # 检查plans文件是否存在
        plans_file = join(preprocessed_folder, f'{plans_identifier}.json')
        if not isfile(plans_file):
            return False

        # 检查每个配置的预处理数据是否存在
        try:
            plans = load_json(plans_file)
            plans_manager = PlansManager(plans)

            for config in configurations:
                if config not in plans_manager.available_configurations:
                    continue  # 跳过不可用的配置

                configuration_manager = plans_manager.get_configuration(config)
                config_folder = join(preprocessed_folder, configuration_manager.data_identifier)

                # 检查配置文件夹是否存在且不为空
                if not isdir(config_folder):
                    return False

                # 检查是否有预处理的数据文件
                npz_files = [f for f in listdir(config_folder) if f.endswith('.npz')]
                if len(npz_files) == 0:
                    return False

        except Exception:
            # 如果读取plans文件或检查过程中出现任何错误，认为预处理未完成
            return False

    return True


def plan_and_preprocess(
    dataset_id: Union[int, List[int]],
    verify_dataset_integrity: bool = False,
    gpu_memory_target: float = 8,
    preprocessor_name: str = 'DefaultPreprocessor',
    overwrite_plans_name: Optional[str] = None,
    overwrite_target_spacing: Optional[List[float]] = None,
    force_target_shape: Optional[List[int]] = None,  # 新增参数：强制指定预处理后的图像大小
    max_batch_size: int = 32,  # 新增参数：最大批次大小限制
    force_n_stages: Optional[int] = None,  # 新增参数：强制指定网络层数
    clean: bool = False,
    configurations: List[str] = ['2d', '3d_fullres', '3d_lowres'],
    num_processes: Optional[List[int]] = None,
    verbose: bool = False,
    force_rerun: bool = False,
) -> Tuple[str, dict]:
    """
    Run fingerprint extraction, experiment planning and preprocessing for the specified dataset(s).

    Args:
        dataset_id: Dataset ID or list of dataset IDs
        verify_dataset_integrity: Set to True to check dataset integrity
        gpu_memory_target: GPU memory target in GB
        preprocessor_name: Name of the preprocessor class to use
        overwrite_plans_name: Custom plans identifier
        overwrite_target_spacing: Custom target spacing for 3d_fullres and 3d_cascade_fullres
        force_target_shape: Force the target shape for preprocessing. 
                           For 2D configurations: can be [x, y] or [z, x, y] (z will be ignored)
                           For 3D configurations: must be [z, x, y]
                           If None, uses automatic spacing calculation
        max_batch_size: Maximum batch size limit to prevent unreasonably large batch sizes
                       Default is 32, can be adjusted based on GPU memory and requirements
        clean: Set to True to overwrite existing fingerprints
        configurations: Configurations for which preprocessing should be run
        num_processes: Number of processes to use for preprocessing
        verbose: Set to True for verbose output
        force_rerun: Set to True to force rerun even if preprocessing is already completed

    Returns:
        Tuple[str, dict]: (plans_identifier, network_configurations)
            - plans_identifier: The identifier of the created plans
            - network_configurations: Dictionary containing detailed network configuration for each configuration
                Format: {
                    'configuration_name': {
                        'architecture': {
                            'network_class_name': str,
                            'n_stages': int,
                            'features_per_stage': List[int],
                            'kernel_sizes': List[List[int]],
                            'strides': List[List[int]],
                            'n_conv_per_stage': List[int],
                            'n_conv_per_stage_decoder': List[int],
                            'conv_op': str,
                            'norm_op': str,
                            'nonlin': str,
                            'conv_bias': bool,
                            'dropout_op': Optional[str]
                        },
                        'data_config': {
                            'batch_size': int,
                            'patch_size': List[int],
                            'spacing': List[float],
                            'median_image_size_in_voxels': List[float]
                        }
                    }
                }
    """
    # 确定plans_identifier
    plans_identifier = 'nnUNetPlans' if overwrite_plans_name is None else overwrite_plans_name

    # 检查预处理是否已经完成
    if not force_rerun and _check_preprocessing_completed(dataset_id, plans_identifier, configurations):
        print("Preprocessing already completed for the specified dataset and configurations. Skipping...")
        if verbose:
            from dinounet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
            if isinstance(dataset_id, int):
                dataset_name = maybe_convert_to_dataset_name(dataset_id)
                print(f"Dataset: {dataset_name}")
            else:
                for did in dataset_id:
                    dataset_name = maybe_convert_to_dataset_name(did)
                    print(f"Dataset: {dataset_name}")
            print(f"Plans identifier: {plans_identifier}")
            print(f"Configurations: {configurations}")

        # 即使跳过预处理，也要提取网络配置信息
        network_configurations = _extract_network_configurations(dataset_id, plans_identifier, configurations)
        return plans_identifier, network_configurations

    if verbose:
        print("Starting plan and preprocess pipeline...")

    # 确保多进程安全
    ctx = multiprocessing.get_context('spawn')

    # 提取指纹
    print("Running fingerprint extraction...")
    p = ctx.Process(target=_run_extract_fingerprints,
                   args=(dataset_id, 'DatasetFingerprintExtractor', 8,
                         verify_dataset_integrity, clean, verbose))
    p.start()
    p.join()

    # 规划实验
    print("Running experiment planning...")
    p = ctx.Process(target=_run_plan_experiments,
                   args=(dataset_id, 'ExperimentPlanner', gpu_memory_target,
                         preprocessor_name, overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages))
    p.start()
    p.join()

    # 设置默认num_processes
    if num_processes is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        num_processes = [default_np[c] if c in default_np.keys() else 4 for c in configurations]

    # 预处理
    print("Running preprocessing...")
    p = ctx.Process(target=_run_preprocess,
                   args=(dataset_id, plans_identifier, configurations, num_processes, verbose))
    p.start()
    p.join()

    # 提取网络配置信息
    network_configurations = _extract_network_configurations(dataset_id, plans_identifier, configurations)

    return plans_identifier, network_configurations


def training(
    dataset_id: Union[int, str],
    configuration: str,
    fold: Union[int, str] = 0,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: Union[torch.device, str] = 'cuda',
    initial_lr: float = None,
    num_epochs: int = None,
    batch_size: int = None
) -> Tuple[str, dict]:
    """
    Run training for the specified dataset, configuration and fold.

    Args:
        dataset_id: Dataset ID or name
        configuration: Configuration to use (e.g. '2d', '3d_fullres')
        fold: Fold to use or 'all' for all folds
        trainer_class: Trainer class to use (name or actual class)
        plans_identifier: Plans identifier
        pretrained_weights: Path to pretrained weights
        num_gpus: Number of GPUs to use
        use_compressed_data: Set to True to use compressed data
        export_validation_probabilities: Set to True to export validation probabilities
        continue_training: Set to True to continue training from checkpoint
        only_run_validation: Set to True to only run validation
        disable_checkpointing: Set to True to disable checkpointing
        val_with_best: Set to True to validate with best checkpoint
        device: Device to use for training
        initial_lr: Initial learning rate (if None, uses default from trainer)
        num_epochs: Number of epochs to train (if None, uses default from trainer)
        batch_size: Batch size for training (if None, uses default from trainer)

    Returns:
        tuple: (output_folder, training_log)
            - output_folder (str): Path to the output folder containing training results
            - training_log (dict): Training log containing epoch-wise metrics with keys:
                - 'epochs': List of epoch numbers
                - 'train_losses': List of training losses per epoch
                - 'val_losses': List of validation losses per epoch
    """
    # 确保导入join函数
    
    # 确保dataset_id是字符串类型（如果是整数）
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)
    
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    
    # 如果设置了自定义参数，我们需要先获取训练器实例，然后修改它
    if initial_lr is not None or num_epochs is not None or batch_size is not None:
        # 检查trainer_class是否为实际的类对象（而不是字符串）
        if not isinstance(trainer_class, str):
            # 如果trainer_class是一个类对象，我们直接使用它
            # 获取preprocessed_dataset_folder
            from dinounet.paths import nnUNet_preprocessed
            dataset_name = maybe_convert_to_dataset_name(dataset_id)
            preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
            
            # 加载plans和dataset_json
            plans_file = join(preprocessed_folder, f"{plans_identifier}.json")
            plans = load_json(plans_file)
            dataset_json_file = join(preprocessed_folder, "dataset.json")
            dataset_json = load_json(dataset_json_file)
            
            # 创建训练器实例
            trainer = trainer_class(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
        else:
            # 如果trainer_class是字符串，使用原有的get_trainer_from_args函数
            trainer = get_trainer_from_args(
                dataset_name_or_id=dataset_id,
                configuration=configuration,
                fold=fold,
                trainer_name=trainer_class,
                plans_identifier=plans_identifier,
                use_compressed=use_compressed_data,
                device=device
            )
        
        # 设置自定义学习率
        if initial_lr is not None:
            trainer.initial_lr = initial_lr
            print(f"Using custom initial learning rate: {initial_lr}")
        
        # 设置自定义训练轮次
        if num_epochs is not None:
            trainer.num_epochs = num_epochs
            print(f"Using custom number of epochs: {num_epochs}")
        
        # 设置自定义批处理大小
        if batch_size is not None:
            # 需要在initialize之前设置，因为initialize会根据batch_size计算其他参数
            if hasattr(trainer, 'batch_size'):
                trainer.batch_size = batch_size
                print(f"Using custom batch size: {batch_size}")
            else:
                print("Warning: Trainer does not have a batch_size attribute. Custom batch size will be ignored.")
        
        # 运行训练
        if pretrained_weights is not None:
            if not trainer.was_initialized:
                trainer.initialize()
            from dinounet.run.load_pretrained_weights import load_pretrained_weights
            load_pretrained_weights(trainer.network, pretrained_weights, verbose=True)
        
        # 设置检查点禁用
        if disable_checkpointing:
            trainer.disable_checkpointing = disable_checkpointing
        
        # 运行训练
        if not only_run_validation:
            trainer.run_training()
        else:
            # 如果只运行验证，需要确保训练器被初始化并加载检查点
            if not trainer.was_initialized:
                trainer.initialize()
            # 加载最终的检查点用于验证
            expected_checkpoint_file = join(trainer.output_folder, 'checkpoint_final.pth')
            if not isfile(expected_checkpoint_file):
                raise RuntimeError(f"Cannot run validation because the training is not finished yet! Expected checkpoint file: {expected_checkpoint_file}")
            trainer.load_checkpoint(expected_checkpoint_file)
        
        # 验证
        if val_with_best:
            trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_best.pth'))
        trainer.perform_actual_validation(export_validation_probabilities)
        
        # 获取输出文件夹和训练日志
        output_folder = trainer.output_folder
        training_log = _extract_training_log(trainer.logger)
    else:
        # 检查trainer_class是否为实际的类对象（而不是字符串）
        if not isinstance(trainer_class, str):
            # 如果是类对象，我们需要创建一个临时模块来存储这个类
            import sys
            import types
            
            # 创建一个临时模块
            mod_name = f"dinounet.training.nnUNetTrainer.{trainer_class.__name__}"
            if mod_name not in sys.modules:
                mod = types.ModuleType(mod_name)
                setattr(mod, trainer_class.__name__, trainer_class)
                sys.modules[mod_name] = mod
            
            # 使用类名作为trainer_class_name
            trainer_class_name = trainer_class.__name__
        else:
            trainer_class_name = trainer_class
        
        # 使用原始的run_training函数
        run_training(
            dataset_name_or_id=dataset_id,
            configuration=configuration,
            fold=fold,
            trainer_class_name=trainer_class_name,
            plans_identifier=plans_identifier,
            pretrained_weights=pretrained_weights,
            num_gpus=num_gpus,
            use_compressed_data=use_compressed_data,
            export_validation_probabilities=export_validation_probabilities,
            continue_training=continue_training,
            only_run_validation=only_run_validation,
            disable_checkpointing=disable_checkpointing,
            val_with_best=val_with_best,
            device=device
        )
        
        # 确定输出文件夹
        from dinounet.paths import nnUNet_results
        output_folder = join(nnUNet_results,
                            f"Dataset{dataset_id}" if isinstance(dataset_id, int) else dataset_id,
                            f"{trainer_class_name}__{plans_identifier}__{configuration}",
                            f"fold_{fold}")

        # 从输出文件夹中加载训练日志
        training_log = _load_training_log_from_folder(output_folder)

    return output_folder, training_log


def evaluate(
    dataset_id: Union[int, str],
    result_folder: str,
    fold: Optional[Union[int, str]] = 0,
    output_file: Optional[str] = None,
    num_processes: int = 8,
    chill: bool = True
) -> dict:
    """
    Evaluate predictions in the result folder.
    
    Args:
        dataset_id: Dataset ID or name
        result_folder: Path to the result folder
        fold: Fold to evaluate (if None, uses the fold from result_folder)
        output_file: Path to output file
        num_processes: Number of processes to use
        chill: Set to True to not crash if folder_pred doesn't have all files
        
    Returns:
        results: Evaluation results
    """
    # 确保dataset_id是字符串类型（如果是整数）
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)
    
    # Convert dataset_id to dataset name if it's an integer
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    
    # Determine fold from result_folder if not provided
    if fold is None and "fold_" in result_folder:
        fold = result_folder.split("fold_")[-1].split("/")[0]
    
    # Get preprocessed dataset folder
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
    
    # Load dataset.json
    dataset_json_file = join(preprocessed_folder, "dataset.json")
    
    # Determine plans file
    plans_identifier = result_folder.split("__")[1]
    plans_file = join(preprocessed_folder, f"{plans_identifier}.json")
    
    # Determine gt and pred folders
    if fold is not None:
        gt_folder = join(preprocessed_folder, "gt_segmentations")
        pred_folder = join(result_folder, "validation")
    else:
        # If no fold is specified, assume we're evaluating test predictions
        gt_folder = join(preprocessed_folder, "gt_segmentations")
        pred_folder = join(result_folder, "test_predictions")
    
    # Run evaluation
    if output_file is None:
        output_file = join(pred_folder, "summary.json")
    
    # 使用单独的进程运行评估，以避免多进程问题
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=compute_metrics_on_folder2,
                   args=(gt_folder, pred_folder, dataset_json_file, plans_file, 
                         output_file, num_processes, chill))
    p.start()
    p.join()
    
    # Load and return results
    from dinounet.evaluation.evaluate_predictions import load_summary_json
    return load_summary_json(output_file) 