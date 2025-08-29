import shutil
from copy import deepcopy
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from dinounet.configuration import ANISO_THRESHOLD
from dinounet.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from dinounet.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from dinounet.paths import nnUNet_raw, nnUNet_preprocessed
from dinounet.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from dinounet.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from dinounet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from dinounet.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from dinounet.utilities.get_network_from_plans import get_network_from_plans
from dinounet.utilities.json_export import recursive_fix_for_json_export
from dinounet.utilities.utils import get_filenames_of_train_images_and_targets


class ExperimentPlanner(object):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 force_target_shape: Union[List[int], Tuple[int, ...]] = None,  # 新增参数：强制指定目标形状
                 max_batch_size: int = 32,  # 新增参数：最大批次大小限制
                 force_n_stages: Optional[int] = None,  # 新增参数：强制指定网络层数
                 suppress_transpose: bool = False):  # 新增参数：使用ImageNet标准化
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.suppress_transpose = suppress_transpose
        self.raw_dataset_folder = join(nnUNet_raw, self.dataset_name)
        preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name)
        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.raw_dataset_folder, self.dataset_json)

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, 'dataset_fingerprint.json')):
            raise RuntimeError('Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint')

        self.dataset_fingerprint = load_json(join(preprocessed_folder, 'dataset_fingerprint.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce nnU-Net v1's configurations as
        # much as possible
        self.UNet_reference_val_3d = 560000000  # 455600128  550000000
        self.UNet_reference_val_2d = 85000000  # 83252480
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320
        self.max_dataset_covered = 0.05 # we limit the batch size so that no more than 5% of the dataset can be seen
        # in a single forward/backward pass

        self.UNet_vram_target_GB = gpu_memory_target_in_gb

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_identifier = plans_name
        self.overwrite_target_spacing = overwrite_target_spacing
        self.force_target_shape = force_target_shape
        self.max_batch_size = max_batch_size
        self.force_n_stages = force_n_stages
        assert overwrite_target_spacing is None or len(overwrite_target_spacing), 'if overwrite_target_spacing is ' \
                                                                                  'used then three floats must be ' \
                                                                                  'given (as list or tuple)'
        assert overwrite_target_spacing is None or all([isinstance(i, float) for i in overwrite_target_spacing]), \
            'if overwrite_target_spacing is used then three floats must be given (as list or tuple)'

        self.plans = None

        if isfile(join(self.raw_dataset_folder, 'splits_final.json')):
            _maybe_copy_splits_file(join(self.raw_dataset_folder, 'splits_final.json'),
                                    join(preprocessed_folder, 'splits_final.json'))

    def determine_reader_writer(self):
        example_image = self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0]
        return determine_reader_writer_from_dataset_json(self.dataset_json, example_image)

    @staticmethod
    def static_estimate_VRAM_usage(patch_size: Tuple[int],
                                   input_channels: int,
                                   output_channels: int,
                                   arch_class_name: str,
                                   arch_kwargs: dict,
                                   arch_kwargs_req_import: Tuple[str, ...]):
        """
        Works for PlainConvUNet, ResidualEncoderUNet
        """
        a = torch.get_num_threads()
        torch.set_num_threads(get_allowed_n_proc_DA())
        # print(f'instantiating network, patch size {patch_size}, pool op: {arch_kwargs["strides"]}')
        net = get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels,
                                     output_channels,
                                     allow_init=False)
        ret = net.compute_conv_feature_map_size(patch_size)
        torch.set_num_threads(a)
        return ret

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_data_or_seg_to_shape
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_shape
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_data_or_seg_to_shape
        resampling_fn_kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs

    def determine_fullres_target_spacing(self, configuration_type: str = '3d') -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        
        Args:
            configuration_type: '2d' or '3d' to specify the configuration type
        """
        if self.overwrite_target_spacing is not None:
            return np.array(self.overwrite_target_spacing)

        # 如果强制指定了目标形状，需要反向计算对应的 spacing
        if self.force_target_shape is not None:
            # 根据原始图像尺寸和强制目标尺寸计算对应的 spacing
            original_spacings = self.dataset_fingerprint['spacings']
            original_shapes = self.dataset_fingerprint['shapes_after_crop']
            
            # 使用中位数作为参考
            median_spacing = np.median(np.vstack(original_spacings), 0)
            median_shape = np.median(np.vstack(original_shapes), 0)
            
            # 根据配置类型处理 force_target_shape
            if configuration_type == '2d':
                # 对于2D配置，只处理x和y维度
                if len(self.force_target_shape) == 2:
                    # 用户只指定了x和y的尺寸
                    target_shape_2d = np.array(self.force_target_shape)
                    median_shape_2d = median_shape[1:]  # 去掉z维度，只取x和y
                    scale_factors_2d = target_shape_2d / median_shape_2d
                    # 正确的spacing计算：spacing变小，图像变大
                    target_spacing_2d = median_spacing[1:] / scale_factors_2d
                    # 对于2D配置，返回2D spacing
                    return target_spacing_2d
                elif len(self.force_target_shape) == 3:
                    # 用户指定了3D尺寸，但对于2D配置，我们只使用x和y
                    target_shape_2d = np.array(self.force_target_shape[1:])  # 只取x和y
                    median_shape_2d = median_shape[1:]  # 去掉z维度
                    scale_factors_2d = target_shape_2d / median_shape_2d
                    # 正确的spacing计算：spacing变小，图像变大
                    target_spacing_2d = median_spacing[1:] / scale_factors_2d
                    # 对于2D配置，返回2D spacing
                    return target_spacing_2d
                else:
                    raise ValueError(f"force_target_shape for 2d configuration should have 2 or 3 elements, got {len(self.force_target_shape)}")
            else:
                # 对于3D配置，处理所有维度
                if len(self.force_target_shape) == 2:
                    # 用户只指定了x和y，z轴保持原始spacing
                    target_shape_2d = np.array(self.force_target_shape)
                    median_shape_2d = median_shape[1:]  # 去掉z维度
                    scale_factors_2d = target_shape_2d / median_shape_2d
                    
                    # 计算x和y轴的spacing（正确的计算）
                    target_spacing_2d = median_spacing[1:] / scale_factors_2d
                    
                    # z轴保持原始spacing
                    target_spacing = np.array([median_spacing[0], target_spacing_2d[0], target_spacing_2d[1]])
                    
                elif len(self.force_target_shape) == 3:
                    target_shape = np.array(self.force_target_shape)
                    scale_factors = target_shape / median_shape
                    # 正确的spacing计算：spacing变小，图像变大
                    target_spacing = median_spacing / scale_factors
                else:
                    raise ValueError(f"force_target_shape for 3d configuration should have 2 or 3 elements, got {len(self.force_target_shape)}")
                
                return target_spacing

        spacings = self.dataset_fingerprint['spacings']
        sizes = self.dataset_fingerprint['shapes_after_crop']

        target = np.percentile(np.vstack(spacings), 50, 0)

        # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        if 'channel_names' not in self.dataset_json.keys():
            print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
                  'enforced soon!')
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']
        normalization_schemes = [get_normalization_scheme(m) for m in modalities.values()]
        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                         normalization_schemes]
        else:
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                                 'True or False and cannot be None'
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_nonzero_mask_for_norm

    def determine_transpose(self):
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        # 根据 force_target_shape 的长度判断配置类型
        if self.force_target_shape is not None:
            if len(self.force_target_shape) == 2:
                # 2D配置
                configuration_type = '2d'
            elif len(self.force_target_shape) == 3:
                # 3D配置
                configuration_type = '3d'
            else:
                # 默认使用3D配置
                configuration_type = '3d'
        else:
            # 默认使用3D配置
            configuration_type = '3d'

        # todo we should use shapes for that as well. Not quite sure how yet
        target_spacing = self.determine_fullres_target_spacing(configuration_type)

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        return transpose_forward, transpose_backward

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict,
                                    override_patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if override_patch_size is not None and len(override_patch_size) == len(spacing):
            initial_patch_size = np.array(list(override_patch_size))
        else:
            if len(spacing) == 3:
                initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
            elif len(spacing) == 2:
                initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
            else:
                raise RuntimeError()

            # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
            # this is different from how nnU-Net v1 does it!
            # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
            initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)
        
        # 如果强制指定了n_stages，则调整网络拓扑
        if self.force_n_stages is not None:
            print(f"强制指定网络层数: {self.force_n_stages}")
            if self.force_n_stages != num_stages:
                print(f"原始层数: {num_stages}, 调整为: {self.force_n_stages}")
                # 重新计算网络拓扑以匹配指定的层数
                network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
                shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                                     self.UNet_featuremap_min_edge_length,
                                                                     self.force_n_stages - 1)  # max_numpool = n_stages - 1
                num_stages = len(pool_op_kernel_sizes)
                print(f"调整后层数: {num_stages}")

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        # we enforce a batch size of at least two, reference values may have been computed for different batch sizes.
        # Correct for that in the while loop if statement
        while (override_patch_size is None) and ((estimate / ref_bs * 2) > reference):
            # print(patch_size, estimate, reference)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            max_numpool_for_recompute = (self.force_n_stages - 1) if self.force_n_stages is not None else 999999
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 max_numpool_for_recompute)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        
        # 添加额外的批次大小限制，防止批次大小过大
        max_reasonable_batch_size = self.max_batch_size  # 使用实例属性
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent, max_reasonable_batch_size), self.UNet_min_batch_size)
        
        # 添加调试信息
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Batch size calculation debug info:")
            print(f"  - reference: {reference}")
            print(f"  - estimate: {estimate}")
            print(f"  - ref_bs: {ref_bs}")
            print(f"  - initial batch_size: {round((reference / estimate) * ref_bs)}")
            print(f"  - bs_corresponding_to_5_percent: {bs_corresponding_to_5_percent}")
            print(f"  - max_reasonable_batch_size: {max_reasonable_batch_size}")
            print(f"  - final batch_size: {batch_size}")
            print(f"  - patch_size: {patch_size}")
            print(f"  - approximate_n_voxels_dataset: {approximate_n_voxels_dataset}")

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan

    def plan_experiment(self):
        """
        MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        """
        # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing('3d')  # 3D配置使用3d参数
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset, _tmp,
                                                               override_patch_size=(
                                                                   np.array(self.force_target_shape)[transpose_forward]
                                                                   if (self.force_target_shape is not None and len(self.force_target_shape) == 3)
                                                                   else None
                                                               ))
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                else:
                    lowres_spacing *= spacing_increase_factor
                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                # print(lowres_spacing)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  tuple([round(i) for i in plan_3d_fullres['spacing'] /
                                                                         lowres_spacing * new_median_shape_transposed]),
                                                                  self.generate_data_identifier('3d_lowres'),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json['numTraining']), _tmp,
                                                                  override_patch_size=None)
                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
            if np.prod(new_median_shape_transposed, dtype=np.float64) / median_num_voxels < 2:
                print(f'Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. '
                      f'3d_fullres: {new_median_shape_transposed}, '
                      f'3d_lowres: {[round(i) for i in plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed]}')
                plan_3d_lowres = None
            if plan_3d_lowres is not None:
                plan_3d_lowres['batch_dice'] = False
                plan_3d_fullres['batch_dice'] = True
            else:
                plan_3d_fullres['batch_dice'] = False
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration - 为2D配置单独计算spacing
        fullres_spacing_2d = self.determine_fullres_target_spacing('2d')  # 2D配置使用2d参数，返回2D spacing
        # 对于2D配置，我们需要构建3D spacing用于计算，z轴保持原始spacing
        median_spacing = np.median(np.vstack(self.dataset_fingerprint['spacings']), 0)
        # z轴保持原始spacing，x和y轴使用计算出的spacing
        fullres_spacing_3d_for_2d = np.array([median_spacing[0], fullres_spacing_2d[0], fullres_spacing_2d[1]])
        fullres_spacing_transposed_2d = fullres_spacing_3d_for_2d[transpose_forward]
        
        # 计算2D配置的新形状
        new_shapes_2d = [compute_new_shape(j, i, fullres_spacing_3d_for_2d) for i, j in
                         zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape_2d = np.median(new_shapes_2d, 0)
        new_median_shape_transposed_2d = new_median_shape_2d[transpose_forward]
        
        # 为2D配置单独计算 approximate_n_voxels_dataset
        approximate_n_voxels_dataset_2d = float(np.prod(new_median_shape_transposed_2d, dtype=np.float64) *
                                               self.dataset_json['numTraining'])
        
        # 如果 force_target_shape 是 2D，则将其视为 2D patch size，按转置轴顺序重排到 2D坐标系（转置后的轴1与轴2）
        override_patch_size_2d = None
        if self.force_target_shape is not None and len(self.force_target_shape) == 2:
            tmp_vec = np.array([1, self.force_target_shape[0], self.force_target_shape[1]])
            tmp_vec_t = tmp_vec[transpose_forward]
            override_patch_size_2d = tmp_vec_t[1:].tolist()

        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed_2d[1:],
                                                   new_median_shape_transposed_2d[1:],
                                                   self.generate_data_identifier('2d'), approximate_n_voxels_dataset_2d,
                                                   _tmp,
                                                   override_patch_size=override_patch_size_2d)
        plan_2d['batch_dice'] = True

        print('2D U-Net configuration:')
        print(plan_2d)
        print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # json is ###. I hate it... "Object of type int64 is not JSON serializable"
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        if plan_3d_lowres is not None:
            plans['configurations']['3d_lowres'] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
            print('3D lowres U-Net configuration:')
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans['configurations']['3d_cascade_fullres'] = {
                    'inherits_from': '3d_fullres',
                    'previous_stage': '3d_lowres'
                }

        self.plans = plans
        self.save_plans(plans)
        return plans

    def save_plans(self, plans):
        recursive_fix_for_json_export(plans)

        plans_file = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')}")

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def load_plans(self, fname: str):
        self.plans = load_json(fname)


def _maybe_copy_splits_file(splits_file: str, target_fname: str):
    if not isfile(target_fname):
        shutil.copy(splits_file, target_fname)
    else:
        # split already exists, do not copy, but check that the splits match.
        # This code allows target_fname to contain more splits than splits_file. This is OK.
        splits_source = load_json(splits_file)
        splits_target = load_json(target_fname)
        # all folds in the source file must match the target file
        for i in range(len(splits_source)):
            train_source = set(splits_source[i]['train'])
            train_target = set(splits_target[i]['train'])
            assert train_target == train_source
            val_source = set(splits_source[i]['val'])
            val_target = set(splits_target[i]['val'])
            assert val_source == val_target


if __name__ == '__main__':
    ExperimentPlanner(2, 8).plan_experiment()
