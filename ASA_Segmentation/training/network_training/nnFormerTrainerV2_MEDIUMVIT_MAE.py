#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture.MEDIUMVIT import MEDIUMVIT
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer import nnFormerTrainer
from nnformer.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


PRETRAIN_PATH = '/data2/huangjunjia/nnFormer/nnFormer_MAE/output/HOG_LOSS/checkpoint-199.pth'


class nnFormerTrainerV2_MEDIUMVIT_MAE(nnFormerTrainer):
    """
    Info for Fabian: same as internal nnFormerTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()
            # self.data_aug_params["do_mirror"] = False
            # self.data_aug_params["do_rotation"] = False

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        self.network = MEDIUMVIT(in_channels=4, out_channels=4, img_size=(128, 128, 128), norm_name='instance',
                                 window_size=32)
        # load encoder
        print("Loading encoder")
        checkpoint = torch.load(PRETRAIN_PATH,
                                map_location='cpu')
        checkpoint_model = checkpoint['model']

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()

        for key in all_keys:
            if key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
                if key.startswith('encoder.vit.patch_embedding.patch_embeddings.1.weight'):
                    new_dict[key[8:]] = checkpoint_model[key].repeat(1, 4)
                elif key.startswith('encoder.vit.blocks') and key.endswith("attn_mask"):
                    new_dict[key[8:]] = checkpoint_model[key].repeat(4, 1, 1)
            else:
                new_dict[key] = checkpoint_model[key]

        checkpoint_model = new_dict
        self.load_state_dict(checkpoint_model)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def load_state_dict(self, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.network, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                self.network.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                self.network.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                self.network.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), self.initial_lr,
                                         weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")
            splits = []
            splits.append(OrderedDict())
            splits[self.fold]['train'] = np.array(
                ['BraTS2021_00795', 'BraTS2021_01267', 'BraTS2021_00442', 'BraTS2021_01108',
                 'BraTS2021_00059', 'BraTS2021_00839', 'BraTS2021_01233', 'BraTS2021_01139',
                 'BraTS2021_00258', 'BraTS2021_00649', 'BraTS2021_01406', 'BraTS2021_00730',
                 'BraTS2021_00221', 'BraTS2021_01526', 'BraTS2021_01029', 'BraTS2021_01271',
                 'BraTS2021_01514', 'BraTS2021_01131', 'BraTS2021_00593', 'BraTS2021_00025',
                 'BraTS2021_01054', 'BraTS2021_01247', 'BraTS2021_01189', 'BraTS2021_00347',
                 'BraTS2021_01624', 'BraTS2021_00402', 'BraTS2021_00602', 'BraTS2021_01357',
                 'BraTS2021_00351', 'BraTS2021_00072', 'BraTS2021_00788', 'BraTS2021_01296',
                 'BraTS2021_01251', 'BraTS2021_01387', 'BraTS2021_01591', 'BraTS2021_01341',
                 'BraTS2021_00436', 'BraTS2021_01535', 'BraTS2021_00657', 'BraTS2021_01491',
                 'BraTS2021_01606', 'BraTS2021_01531', 'BraTS2021_00433', 'BraTS2021_00058',
                 'BraTS2021_01077', 'BraTS2021_01511', 'BraTS2021_01226', 'BraTS2021_01395',
                 'BraTS2021_01224', 'BraTS2021_01513', 'BraTS2021_01652', 'BraTS2021_01575',
                 'BraTS2021_00477', 'BraTS2021_01052', 'BraTS2021_01064', 'BraTS2021_01179',
                 'BraTS2021_01031', 'BraTS2021_00303', 'BraTS2021_00441', 'BraTS2021_01213',
                 'BraTS2021_01534', 'BraTS2021_01190', 'BraTS2021_00683', 'BraTS2021_00618',
                 'BraTS2021_00096', 'BraTS2021_00107', 'BraTS2021_01066', 'BraTS2021_01091',
                 'BraTS2021_00470', 'BraTS2021_01464', 'BraTS2021_00459', 'BraTS2021_00014',
                 'BraTS2021_01584', 'BraTS2021_01356', 'BraTS2021_00136', 'BraTS2021_00698',
                 'BraTS2021_01058', 'BraTS2021_01048', 'BraTS2021_01171', 'BraTS2021_00691',
                 'BraTS2021_00176', 'BraTS2021_01364', 'BraTS2021_01101', 'BraTS2021_00132',
                 'BraTS2021_01235', 'BraTS2021_01461', 'BraTS2021_00787', 'BraTS2021_01013',
                 'BraTS2021_01566', 'BraTS2021_00162', 'BraTS2021_01105', 'BraTS2021_01619',
                 'BraTS2021_01015', 'BraTS2021_01289', 'BraTS2021_01405', 'BraTS2021_01552',
                 'BraTS2021_01176', 'BraTS2021_01124', 'BraTS2021_00715', 'BraTS2021_00791',
                 'BraTS2021_00028', 'BraTS2021_00642', 'BraTS2021_00590', 'BraTS2021_00187',
                 'BraTS2021_00328', 'BraTS2021_01326', 'BraTS2021_01479', 'BraTS2021_01115',
                 'BraTS2021_00519', 'BraTS2021_01297', 'BraTS2021_00440', 'BraTS2021_00350',
                 'BraTS2021_01487', 'BraTS2021_01248', 'BraTS2021_01337', 'BraTS2021_00481',
                 'BraTS2021_00241', 'BraTS2021_01144', 'BraTS2021_01148', 'BraTS2021_00656',
                 'BraTS2021_01193', 'BraTS2021_01415', 'BraTS2021_00443', 'BraTS2021_01218',
                 'BraTS2021_01025', 'BraTS2021_01477', 'BraTS2021_01448', 'BraTS2021_00542',
                 'BraTS2021_00806', 'BraTS2021_01119', 'BraTS2021_00418', 'BraTS2021_01458',
                 'BraTS2021_01605', 'BraTS2021_01370', 'BraTS2021_01243', 'BraTS2021_01361',
                 'BraTS2021_01554', 'BraTS2021_00685', 'BraTS2021_00444', 'BraTS2021_00625',
                 'BraTS2021_01620', 'BraTS2021_00095', 'BraTS2021_00780', 'BraTS2021_00298',
                 'BraTS2021_01573', 'BraTS2021_00416', 'BraTS2021_01203', 'BraTS2021_00611',
                 'BraTS2021_01588', 'BraTS2021_01076', 'BraTS2021_00792', 'BraTS2021_01019',
                 'BraTS2021_00376', 'BraTS2021_00724', 'BraTS2021_00756', 'BraTS2021_01548',
                 'BraTS2021_01026', 'BraTS2021_01079', 'BraTS2021_00309', 'BraTS2021_00254',
                 'BraTS2021_01443', 'BraTS2021_00300', 'BraTS2021_00661', 'BraTS2021_01347',
                 'BraTS2021_01475', 'BraTS2021_01229', 'BraTS2021_00138', 'BraTS2021_01009',
                 'BraTS2021_00610', 'BraTS2021_01570', 'BraTS2021_01020', 'BraTS2021_01617',
                 'BraTS2021_01454', 'BraTS2021_00156', 'BraTS2021_01465', 'BraTS2021_01642',
                 'BraTS2021_00727', 'BraTS2021_00767', 'BraTS2021_00242', 'BraTS2021_01202',
                 'BraTS2021_00737', 'BraTS2021_01039', 'BraTS2021_01378', 'BraTS2021_00740',
                 'BraTS2021_01093', 'BraTS2021_01122', 'BraTS2021_01317', 'BraTS2021_00801',
                 'BraTS2021_00192', 'BraTS2021_01165', 'BraTS2021_00375', 'BraTS2021_00809',
                 'BraTS2021_00120', 'BraTS2021_00134', 'BraTS2021_01325', 'BraTS2021_01059',
                 'BraTS2021_00088', 'BraTS2021_01240', 'BraTS2021_00708', 'BraTS2021_01452',
                 'BraTS2021_01343', 'BraTS2021_00261', 'BraTS2021_01485', 'BraTS2021_00054',
                 'BraTS2021_00106', 'BraTS2021_00659', 'BraTS2021_01175', 'BraTS2021_01432',
                 'BraTS2021_01078', 'BraTS2021_01425', 'BraTS2021_00619', 'BraTS2021_00544',
                 'BraTS2021_00167', 'BraTS2021_00571', 'BraTS2021_00690', 'BraTS2021_01600',
                 'BraTS2021_00341', 'BraTS2021_01587', 'BraTS2021_01060', 'BraTS2021_00243',
                 'BraTS2021_01136', 'BraTS2021_00498', 'BraTS2021_01246', 'BraTS2021_00364', 'BraTS2021_00053',
                 'BraTS2021_01146', 'BraTS2021_01656', 'BraTS2021_01186', 'BraTS2021_00811', 'BraTS2021_00019',
                 'BraTS2021_01389', 'BraTS2021_01645', 'BraTS2021_00171', 'BraTS2021_01339', 'BraTS2021_00655',
                 'BraTS2021_00547', 'BraTS2021_00557', 'BraTS2021_00684', 'BraTS2021_01301', 'BraTS2021_01660',
                 'BraTS2021_01647', 'BraTS2021_00576', 'BraTS2021_01022', 'BraTS2021_01132', 'BraTS2021_01088',
                 'BraTS2021_00022', 'BraTS2021_01208', 'BraTS2021_00739', 'BraTS2021_00559', 'BraTS2021_01116',
                 'BraTS2021_01260', 'BraTS2021_00273', 'BraTS2021_00081', 'BraTS2021_00567', 'BraTS2021_01034',
                 'BraTS2021_00110', 'BraTS2021_01114', 'BraTS2021_00837', 'BraTS2021_01408', 'BraTS2021_01409',
                 'BraTS2021_01505', 'BraTS2021_01302', 'BraTS2021_01401', 'BraTS2021_00823', 'BraTS2021_00429',
                 'BraTS2021_01135', 'BraTS2021_00137', 'BraTS2021_01450', 'BraTS2021_00757', 'BraTS2021_00391',
                 'BraTS2021_01336', 'BraTS2021_00294', 'BraTS2021_01551', 'BraTS2021_00150', 'BraTS2021_00324',
                 'BraTS2021_00530', 'BraTS2021_01313', 'BraTS2021_01596', 'BraTS2021_01192', 'BraTS2021_01530',
                 'BraTS2021_01363', 'BraTS2021_00143', 'BraTS2021_00269', 'BraTS2021_00228', 'BraTS2021_01182',
                 'BraTS2021_01303', 'BraTS2021_01499', 'BraTS2021_01127', 'BraTS2021_01206', 'BraTS2021_01104',
                 'BraTS2021_01519', 'BraTS2021_00360', 'BraTS2021_01085', 'BraTS2021_01446', 'BraTS2021_01572',
                 'BraTS2021_00621', 'BraTS2021_00688', 'BraTS2021_00704', 'BraTS2021_00675', 'BraTS2021_00128',
                 'BraTS2021_00126', 'BraTS2021_00033', 'BraTS2021_01659', 'BraTS2021_01366', 'BraTS2021_01241',
                 'BraTS2021_01166', 'BraTS2021_01665', 'BraTS2021_01057', 'BraTS2021_00194', 'BraTS2021_00532',
                 'BraTS2021_00840', 'BraTS2021_01014', 'BraTS2021_01323', 'BraTS2021_00505', 'BraTS2021_01459',
                 'BraTS2021_00581', 'BraTS2021_01120', 'BraTS2021_01220', 'BraTS2021_01170', 'BraTS2021_00158',
                 'BraTS2021_01609', 'BraTS2021_00679', 'BraTS2021_01655', 'BraTS2021_00154', 'BraTS2021_01304',
                 'BraTS2021_00426', 'BraTS2021_00735', 'BraTS2021_01183', 'BraTS2021_01641', 'BraTS2021_01017',
                 'BraTS2021_00784', 'BraTS2021_00206', 'BraTS2021_01173', 'BraTS2021_01621', 'BraTS2021_01096',
                 'BraTS2021_00045', 'BraTS2021_01560', 'BraTS2021_00652', 'BraTS2021_00222', 'BraTS2021_00578',
                 'BraTS2021_00066', 'BraTS2021_00016', 'BraTS2021_01421', 'BraTS2021_01315', 'BraTS2021_00251',
                 'BraTS2021_00102', 'BraTS2021_01559', 'BraTS2021_01346', 'BraTS2021_01498', 'BraTS2021_01153',
                 'BraTS2021_01074', 'BraTS2021_01335', 'BraTS2021_00538', 'BraTS2021_01311', 'BraTS2021_01637',
                 'BraTS2021_01351', 'BraTS2021_00500', 'BraTS2021_01462', 'BraTS2021_01312', 'BraTS2021_00009',
                 'BraTS2021_01016', 'BraTS2021_00052', 'BraTS2021_00286', 'BraTS2021_01024', 'BraTS2021_01143',
                 'BraTS2021_01541', 'BraTS2021_00048', 'BraTS2021_01444', 'BraTS2021_00658', 'BraTS2021_00723',
                 'BraTS2021_01441', 'BraTS2021_01125', 'BraTS2021_00800', 'BraTS2021_00405', 'BraTS2021_01222',
                 'BraTS2021_01040', 'BraTS2021_01557', 'BraTS2021_00523', 'BraTS2021_01654', 'BraTS2021_00388',
                 'BraTS2021_00606', 'BraTS2021_00382', 'BraTS2021_01657', 'BraTS2021_01180', 'BraTS2021_00716',
                 'BraTS2021_01268', 'BraTS2021_00412', 'BraTS2021_00615', 'BraTS2021_01061', 'BraTS2021_01250',
                 'BraTS2021_00097', 'BraTS2021_00201', 'BraTS2021_00468', 'BraTS2021_00623', 'BraTS2021_01216',
                 'BraTS2021_00094', 'BraTS2021_01021', 'BraTS2021_00636', 'BraTS2021_01612', 'BraTS2021_01416',
                 'BraTS2021_01492', 'BraTS2021_01273', 'BraTS2021_01187', 'BraTS2021_00348', 'BraTS2021_01377',
                 'BraTS2021_01223', 'BraTS2021_00594', 'BraTS2021_00217', 'BraTS2021_00999', 'BraTS2021_00227',
                 'BraTS2021_00452', 'BraTS2021_01259', 'BraTS2021_00810', 'BraTS2021_00032', 'BraTS2021_00373',
                 'BraTS2021_01035', 'BraTS2021_01200', 'BraTS2021_00577', 'BraTS2021_01242', 'BraTS2021_01023',
                 'BraTS2021_01252', 'BraTS2021_01344', 'BraTS2021_01068', 'BraTS2021_01160', 'BraTS2021_01512',
                 'BraTS2021_01209', 'BraTS2021_00480', 'BraTS2021_00170', 'BraTS2021_00525', 'BraTS2021_00586',
                 'BraTS2021_01199', 'BraTS2021_00582', 'BraTS2021_00216', 'BraTS2021_00401', 'BraTS2021_00598',
                 'BraTS2021_00188', 'BraTS2021_01134', 'BraTS2021_01128', 'BraTS2021_01314', 'BraTS2021_00285',
                 'BraTS2021_00570', 'BraTS2021_00667', 'BraTS2021_00061', 'BraTS2021_01261', 'BraTS2021_00680',
                 'BraTS2021_00127', 'BraTS2021_00089', 'BraTS2021_01643', 'BraTS2021_00063', 'BraTS2021_01520',
                 'BraTS2021_00693', 'BraTS2021_01329', 'BraTS2021_00565', 'BraTS2021_01590', 'BraTS2021_00111',
                 'BraTS2021_00734', 'BraTS2021_00624', 'BraTS2021_01292', 'BraTS2021_01568', 'BraTS2021_01156',
                 'BraTS2021_01478', 'BraTS2021_00677', 'BraTS2021_00305', 'BraTS2021_00563', 'BraTS2021_01410',
                 'BraTS2021_00379', 'BraTS2021_01403', 'BraTS2021_01556', 'BraTS2021_00640', 'BraTS2021_00349',
                 'BraTS2021_00036', 'BraTS2021_00510', 'BraTS2021_01284', 'BraTS2021_00575', 'BraTS2021_00607',
                 'BraTS2021_00549', 'BraTS2021_01495', 'BraTS2021_00231', 'BraTS2021_00430', 'BraTS2021_00746',
                 'BraTS2021_00714', 'BraTS2021_01072', 'BraTS2021_01095', 'BraTS2021_00796', 'BraTS2021_01502',
                 'BraTS2021_00820', 'BraTS2021_01651', 'BraTS2021_00291', 'BraTS2021_00511', 'BraTS2021_01214',
                 'BraTS2021_01046', 'BraTS2021_01278', 'BraTS2021_00495', 'BraTS2021_00469', 'BraTS2021_01368',
                 'BraTS2021_01005', 'BraTS2021_00219', 'BraTS2021_00777', 'BraTS2021_00773', 'BraTS2021_00218',
                 'BraTS2021_01527', 'BraTS2021_01417', 'BraTS2021_01438', 'BraTS2021_00818', 'BraTS2021_00301',
                 'BraTS2021_00419', 'BraTS2021_00676', 'BraTS2021_00587', 'BraTS2021_00325', 'BraTS2021_01576',
                 'BraTS2021_00340', 'BraTS2021_01219', 'BraTS2021_00253', 'BraTS2021_01420', 'BraTS2021_00453',
                 'BraTS2021_01286', 'BraTS2021_01585', 'BraTS2021_01308', 'BraTS2021_00789', 'BraTS2021_01130',
                 'BraTS2021_01476', 'BraTS2021_00451', 'BraTS2021_01047', 'BraTS2021_00237', 'BraTS2021_01320',
                 'BraTS2021_01069', 'BraTS2021_01129', 'BraTS2021_01316', 'BraTS2021_01003', 'BraTS2021_01295',
                 'BraTS2021_00526', 'BraTS2021_01633', 'BraTS2021_00555', 'BraTS2021_01349', 'BraTS2021_01002',
                 'BraTS2021_00056', 'BraTS2021_00751', 'BraTS2021_00742', 'BraTS2021_01154', 'BraTS2021_01181',
                 'BraTS2021_00506', 'BraTS2021_01230', 'BraTS2021_00006', 'BraTS2021_01553', 'BraTS2021_01369',
                 'BraTS2021_01331', 'BraTS2021_00395', 'BraTS2021_00021', 'BraTS2021_00246', 'BraTS2021_00370',
                 'BraTS2021_01263', 'BraTS2021_00149', 'BraTS2021_00321', 'BraTS2021_01319', 'BraTS2021_01118',
                 'BraTS2021_00239', 'BraTS2021_01390', 'BraTS2021_01178', 'BraTS2021_01460', 'BraTS2021_01291',
                 'BraTS2021_01493', 'BraTS2021_00601', 'BraTS2021_01262', 'BraTS2021_00520', 'BraTS2021_00750',
                 'BraTS2021_01404', 'BraTS2021_01435', 'BraTS2021_01283', 'BraTS2021_00651', 'BraTS2021_00718',
                 'BraTS2021_00613', 'BraTS2021_01121', 'BraTS2021_00304', 'BraTS2021_01062', 'BraTS2021_00151',
                 'BraTS2021_00819', 'BraTS2021_00185', 'BraTS2021_01354', 'BraTS2021_00803', 'BraTS2021_01269',
                 'BraTS2021_00533', 'BraTS2021_00259', 'BraTS2021_01545', 'BraTS2021_00612', 'BraTS2021_00550',
                 'BraTS2021_00816', 'BraTS2021_01629', 'BraTS2021_00322', 'BraTS2021_00528', 'BraTS2021_00183',
                 'BraTS2021_00299', 'BraTS2021_01028', 'BraTS2021_01564', 'BraTS2021_00728', 'BraTS2021_00545',
                 'BraTS2021_01607', 'BraTS2021_01582', 'BraTS2021_00386', 'BraTS2021_00031', 'BraTS2021_00233',
                 'BraTS2021_01594', 'BraTS2021_01543', 'BraTS2021_00729', 'BraTS2021_00283', 'BraTS2021_00212',
                 'BraTS2021_01184', 'BraTS2021_00838', 'BraTS2021_00572', 'BraTS2021_00518', 'BraTS2021_01164',
                 'BraTS2021_01207', 'BraTS2021_01352', 'BraTS2021_00782', 'BraTS2021_00654', 'BraTS2021_01152',
                 'BraTS2021_01504', 'BraTS2021_00397', 'BraTS2021_01532', 'BraTS2021_00352', 'BraTS2021_00236',
                 'BraTS2021_00479', 'BraTS2021_00207', 'BraTS2021_00026', 'BraTS2021_00496', 'BraTS2021_01488',
                 'BraTS2021_00144', 'BraTS2021_00753', 'BraTS2021_01618', 'BraTS2021_01008', 'BraTS2021_01626',
                 'BraTS2021_01373', 'BraTS2021_00249', 'BraTS2021_00732', 'BraTS2021_01631', 'BraTS2021_00311',
                 'BraTS2021_00103', 'BraTS2021_00758', 'BraTS2021_00483', 'BraTS2021_00105', 'BraTS2021_01345',
                 'BraTS2021_00140', 'BraTS2021_01407', 'BraTS2021_00024', 'BraTS2021_01398', 'BraTS2021_01106',
                 'BraTS2021_00406', 'BraTS2021_01561', 'BraTS2021_00399', 'BraTS2021_00165', 'BraTS2021_01442',
                 'BraTS2021_00616', 'BraTS2021_01126', 'BraTS2021_00296', 'BraTS2021_01603', 'BraTS2021_00425',
                 'BraTS2021_01521', 'BraTS2021_01518', 'BraTS2021_00152', 'BraTS2021_00568', 'BraTS2021_00124',
                 'BraTS2021_01138', 'BraTS2021_00596', 'BraTS2021_01396', 'BraTS2021_01099', 'BraTS2021_01385',
                 'BraTS2021_00002', 'BraTS2021_01196', 'BraTS2021_01358', 'BraTS2021_01583', 'BraTS2021_01334',
                 'BraTS2021_01007', 'BraTS2021_00148', 'BraTS2021_01615', 'BraTS2021_00366', 'BraTS2021_00682',
                 'BraTS2021_00100', 'BraTS2021_00414', 'BraTS2021_01043', 'BraTS2021_01236', 'BraTS2021_00507',
                 'BraTS2021_00011', 'BraTS2021_01616', 'BraTS2021_01437', 'BraTS2021_00504', 'BraTS2021_01436',
                 'BraTS2021_01201', 'BraTS2021_00454', 'BraTS2021_01280', 'BraTS2021_00099', 'BraTS2021_00824',
                 'BraTS2021_00281', 'BraTS2021_01157', 'BraTS2021_01439', 'BraTS2021_01549', 'BraTS2021_01234',
                 'BraTS2021_01001', 'BraTS2021_01508', 'BraTS2021_01524', 'BraTS2021_00317', 'BraTS2021_01290',
                 'BraTS2021_01162', 'BraTS2021_01073', 'BraTS2021_01362', 'BraTS2021_01237', 'BraTS2021_01468',
                 'BraTS2021_01228', 'BraTS2021_00725', 'BraTS2021_00626', 'BraTS2021_00060', 'BraTS2021_00380',
                 'BraTS2021_00588', 'BraTS2021_01451', 'BraTS2021_01494', 'BraTS2021_01422', 'BraTS2021_00078',
                 'BraTS2021_00744', 'BraTS2021_00118', 'BraTS2021_01318', 'BraTS2021_01298', 'BraTS2021_00267',
                 'BraTS2021_01244', 'BraTS2021_01275', 'BraTS2021_00668', 'BraTS2021_01380', 'BraTS2021_01081',
                 'BraTS2021_00772', 'BraTS2021_01636', 'BraTS2021_01299', 'BraTS2021_01041', 'BraTS2021_01087',
                 'BraTS2021_01049', 'BraTS2021_00062', 'BraTS2021_00389', 'BraTS2021_01276', 'BraTS2021_00516',
                 'BraTS2021_01265', 'BraTS2021_00431', 'BraTS2021_01569', 'BraTS2021_01294', 'BraTS2021_00830',
                 'BraTS2021_00210', 'BraTS2021_00387', 'BraTS2021_01168', 'BraTS2021_00186', 'BraTS2021_01516',
                 'BraTS2021_01086', 'BraTS2021_01455', 'BraTS2021_00211', 'BraTS2021_01340', 'BraTS2021_00802',
                 'BraTS2021_01662', 'BraTS2021_00605', 'BraTS2021_01249', 'BraTS2021_00282', 'BraTS2021_01453',
                 'BraTS2021_01429', 'BraTS2021_01365', 'BraTS2021_01507', 'BraTS2021_00012', 'BraTS2021_01011',
                 'BraTS2021_01141', 'BraTS2021_01371', 'BraTS2021_00446', 'BraTS2021_01045', 'BraTS2021_00834',
                 'BraTS2021_01471', 'BraTS2021_01350', 'BraTS2021_00367', 'BraTS2021_00400', 'BraTS2021_00018',
                 'BraTS2021_01386', 'BraTS2021_01376', 'BraTS2021_01307', 'BraTS2021_00512', 'BraTS2021_01185',
                 'BraTS2021_00297', 'BraTS2021_01486', 'BraTS2021_00074', 'BraTS2021_01367', 'BraTS2021_00334',
                 'BraTS2021_01050', 'BraTS2021_00196', 'BraTS2021_01360', 'BraTS2021_00274', 'BraTS2021_00449',
                 'BraTS2021_00409', 'BraTS2021_01374', 'BraTS2021_00070', 'BraTS2021_01650', 'BraTS2021_00501',
                 'BraTS2021_01592', 'BraTS2021_01628', 'BraTS2021_00090', 'BraTS2021_01467', 'BraTS2021_01075',
                 'BraTS2021_01632', 'BraTS2021_01231', 'BraTS2021_01635', 'BraTS2021_00121', 'BraTS2021_01481',
                 'BraTS2021_01169', 'BraTS2021_01608', 'BraTS2021_01145', 'BraTS2021_01270', 'BraTS2021_01070',
                 'BraTS2021_00071', 'BraTS2021_00157', 'BraTS2021_01089', 'BraTS2021_00262', 'BraTS2021_01538',
                 'BraTS2021_01480', 'BraTS2021_01400', 'BraTS2021_01044', 'BraTS2021_01558', 'BraTS2021_00270',
                 'BraTS2021_01661', 'BraTS2021_00663', 'BraTS2021_00339', 'BraTS2021_01427', 'BraTS2021_00195',
                 'BraTS2021_00478', 'BraTS2021_01285', 'BraTS2021_00008', 'BraTS2021_00112', 'BraTS2021_00104',
                 'BraTS2021_00599', 'BraTS2021_01348', 'BraTS2021_01433', 'BraTS2021_00524', 'BraTS2021_00314',
                 'BraTS2021_01466', 'BraTS2021_00774', 'BraTS2021_00191', 'BraTS2021_00781', 'BraTS2021_01581',
                 'BraTS2021_01172', 'BraTS2021_01163', 'BraTS2021_01205', 'BraTS2021_00805', 'BraTS2021_00517',
                 'BraTS2021_00646', 'BraTS2021_01155', 'BraTS2021_00736', 'BraTS2021_01194', 'BraTS2021_01589',
                 'BraTS2021_00284', 'BraTS2021_00159', 'BraTS2021_01497', 'BraTS2021_01293', 'BraTS2021_00703',
                 'BraTS2021_01253', 'BraTS2021_01482', 'BraTS2021_00731', 'BraTS2021_01279', 'BraTS2021_01469',
                 'BraTS2021_01653', 'BraTS2021_01281', 'BraTS2021_01394', 'BraTS2021_00413', 'BraTS2021_00147',
                 'BraTS2021_00030', 'BraTS2021_01414', 'BraTS2021_01355', 'BraTS2021_00250', 'BraTS2021_00077',
                 'BraTS2021_00579', 'BraTS2021_01563', 'BraTS2021_01084', 'BraTS2021_01306', 'BraTS2021_00122',
                 'BraTS2021_00448', 'BraTS2021_01383', 'BraTS2021_00327', 'BraTS2021_00049', 'BraTS2021_00706'])
            splits[self.fold]['val'] = np.array(
                ['BraTS2021_00290', 'BraTS2021_01300', 'BraTS2021_00491', 'BraTS2021_01574', 'BraTS2021_01542',
                 'BraTS2021_01195', 'BraTS2021_01473', 'BraTS2021_00797', 'BraTS2021_00760', 'BraTS2021_01342',
                 'BraTS2021_01555', 'BraTS2021_01158', 'BraTS2021_01445', 'BraTS2021_00329', 'BraTS2021_01256',
                 'BraTS2021_01423', 'BraTS2021_00494', 'BraTS2021_01257', 'BraTS2021_00316', 'BraTS2021_01282',
                 'BraTS2021_00650', 'BraTS2021_01221', 'BraTS2021_00537', 'BraTS2021_00312', 'BraTS2021_00631',
                 'BraTS2021_01215', 'BraTS2021_01393', 'BraTS2021_00597', 'BraTS2021_00369', 'BraTS2021_00085',
                 'BraTS2021_00622', 'BraTS2021_01109', 'BraTS2021_01598', 'BraTS2021_00146', 'BraTS2021_01496',
                 'BraTS2021_00464', 'BraTS2021_00109', 'BraTS2021_01113', 'BraTS2021_00392', 'BraTS2021_00017',
                 'BraTS2021_01434', 'BraTS2021_01212', 'BraTS2021_00493', 'BraTS2021_01648', 'BraTS2021_00343',
                 'BraTS2021_00775', 'BraTS2021_01593', 'BraTS2021_01627', 'BraTS2021_00689', 'BraTS2021_01197',
                 'BraTS2021_00116', 'BraTS2021_01161', 'BraTS2021_01167', 'BraTS2021_01579', 'BraTS2021_01546',
                 'BraTS2021_01359', 'BraTS2021_00674', 'BraTS2021_01503', 'BraTS2021_01324', 'BraTS2021_00310',
                 'BraTS2021_00407', 'BraTS2021_01067', 'BraTS2021_01623', 'BraTS2021_00417', 'BraTS2021_00584',
                 'BraTS2021_01332', 'BraTS2021_01634', 'BraTS2021_00574', 'BraTS2021_01571', 'BraTS2021_01111',
                 'BraTS2021_00155', 'BraTS2021_01440', 'BraTS2021_01123', 'BraTS2021_01159', 'BraTS2021_00539',
                 'BraTS2021_00513', 'BraTS2021_01030', 'BraTS2021_00193', 'BraTS2021_01112', 'BraTS2021_00747',
                 'BraTS2021_01027', 'BraTS2021_01238', 'BraTS2021_01431', 'BraTS2021_01639', 'BraTS2021_00247',
                 'BraTS2021_01094', 'BraTS2021_00087', 'BraTS2021_00485', 'BraTS2021_00320', 'BraTS2021_01602',
                 'BraTS2021_01227', 'BraTS2021_01664', 'BraTS2021_01090', 'BraTS2021_01254', 'BraTS2021_01399',
                 'BraTS2021_00346', 'BraTS2021_01622', 'BraTS2021_01198', 'BraTS2021_00371', 'BraTS2021_01489',
                 'BraTS2021_00204', 'BraTS2021_00814', 'BraTS2021_00318', 'BraTS2021_01056', 'BraTS2021_00220',
                 'BraTS2021_01517', 'BraTS2021_01638', 'BraTS2021_01225', 'BraTS2021_00836', 'BraTS2021_00687',
                 'BraTS2021_00768', 'BraTS2021_01384', 'BraTS2021_01375', 'BraTS2021_01604', 'BraTS2021_01000',
                 'BraTS2021_00804', 'BraTS2021_01142', 'BraTS2021_01614', 'BraTS2021_01018', 'BraTS2021_00123',
                 'BraTS2021_00793', 'BraTS2021_01474', 'BraTS2021_01533', 'BraTS2021_01036', 'BraTS2021_01430'])
            save_pickle(splits, splits_file)
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']
            self.print_to_log_file("This split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        print(self.deep_supervision_scales)

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnFormerTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
