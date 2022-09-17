from collections import OrderedDict
import numpy as np
import torch.utils.data as data
from multiprocessing import Pool
from nnformer.configuration import default_num_threads
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
import torch
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from timm.models.layers import drop_path, to_2tuple, trunc_normal_, to_3tuple
from nnformer.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from batchgenerators.transforms.sample_normalization_transforms import RangeTransform


def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def setup_DA_params():
    data_aug_params = default_3D_augmentation_params
    # data_aug_params['rotation_x'] = (0, 0)
    # data_aug_params['rotation_y'] = (0, 0)
    # data_aug_params['rotation_z'] = (0, 0)
    #
    data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    patch_size = [128, 128, 128]

    patch_size_for_spatialtransform = patch_size

    data_aug_params["scale_range"] = (0.7, 1.4)
    data_aug_params["do_elastic"] = False
    data_aug_params['selected_seg_channels'] = [0]
    data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

    data_aug_params["num_cached_per_thread"] = 2

    data_aug_params["num_cached_per_thread"] = 2

    return data_aug_params


class DataLoader3D(data.Dataset):
    def __init__(self, data_path, crop_size, window_size, mask_ratio, transform):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        self.window_size = to_3tuple(window_size)
        self.mask_ratio = mask_ratio
        self.transform = transform

        l, w, h = self.window_size
        self.num_windows = l * w * h
        self.num_mask = int(mask_ratio * self.num_windows)

        self.dataset = self.load_dataset()
        self.list_of_keys = list(self.dataset.keys())

    def load_dataset(self):
        print("loading dataset")
        case_identifiers = get_case_identifiers(self.data_path)
        case_identifiers.sort()
        dataset = OrderedDict()
        for c in case_identifiers:
            # if c.startswith('sub-OAS'):
            dataset[c] = OrderedDict()
            dataset[c]['data_file'] = join(self.data_path, "%s.npz" % c)
            dataset[c]['properties_file'] = join(self.data_path, "%s.pkl" % c)

        return dataset

    def __len__(self):
        return len(self.list_of_keys)

    def __getitem__(self, item):
        mask = np.hstack([
            np.zeros(self.num_windows - self.num_mask),
            np.ones(self.num_mask)
        ])
        np.random.shuffle(mask)

        key = self.list_of_keys[item]

        data_all = np.load(self.dataset[key]['data_file'][:-4] + ".npz", 'r')["data"]
        data = data_all[:-1][np.newaxis, ::]
        seg = data_all[-1:][np.newaxis, ::]
        tr_data = {'data': data, 'seg': seg, 'mask': mask}
        if self.transform:
            tr_data = self.transform(**tr_data)
        tr_data["data"] = tr_data["data"][0, ::]
        tr_data["seg"] = tr_data["seg"][0, ::]
        tr_data['key'] = key
        return tr_data


def get_transform():
    params = setup_DA_params()
    tr_transforms = []
    tr_transforms.append(SpatialTransform(
        params['patch_size_for_spatialtransform'], patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(NumpyToTensor(['data', 'mask'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def crop_only_tran():
    params = setup_DA_params()
    tr_transforms = []
    tr_transforms.append(SpatialTransform(
        params['patch_size_for_spatialtransform'], patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=False, angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=False, scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    tr_transforms.append(NumpyToTensor(['data', 'seg', 'mask'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def build_pretraining_dataset(args):
    if args.transform:
        transform = get_transform()
    else:
        transform = Compose([NumpyToTensor(['data', 'seg', 'mask'], 'float')])
    print("Data Aug = %s" % str(transform))
    dl = DataLoader3D(
        data_path=args.data_path,
        crop_size=args.input_size,
        window_size=args.window_size,
        mask_ratio=args.mask_ratio,
        transform=transform
    )
    return dl


if __name__ == '__main__':
    transform = crop_only_tran()

    dl = DataLoader3D(
        "/data2/huangjunjia/nnFormer/nnFormer_preprocessed/Task777_MAE/nnFormerData_plans_v2.1_stage0",
        [128, 128, 128],
        [16, 16, 16],
        0.75,
        transform
    )

    save_path = "/data2/huangjunjia/nnFormer/nnFormer_preprocessed/Task777_MAE/DATA_AFTERTRANS"
    maybe_mkdir_p(save_path)
    sampler_train = torch.utils.data.RandomSampler(dl)
    data_loader_train = torch.utils.data.DataLoader(
        dl, sampler=sampler_train,
        batch_size=12,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    import matplotlib.pyplot as plt

    for bat in data_loader_train:
        for sub_data, sub_seg, sub_key in zip(bat['data'], bat['seg'], bat['key']):
            # print(sub_data.shape)
            # plt.imshow(sub_data.numpy()[0, 64, ::], cmap='gray')
            # plt.show()
            # plt.imshow(sub_seg.numpy()[0, 64, ::], cmap='gray')
            # plt.show()
            sub_save = np.vstack((sub_data.numpy(), sub_seg.numpy()))
            # print(sub_save.shape)
            print("Saving file: ", sub_key)
            np.savez(join(save_path, sub_key + ".npz"), data=sub_save)
            # break
        # break
        # mask = np.ones((img_shape))
        # sub_save = np.vstack((sub, mask))
        # print("Saving file : ", key)
        # np.savez(join(save_path, key+".npz"), data=sub_save)
    # break
    #
    # ori_path = "/data2/huangjunjia/nnFormer/nnFormer_preprocessed/Task888_PretrainData/DATA_AFTERTRANS"
    # des_path = "/data2/huangjunjia/nnFormer/nnFormer_preprocessed/Task888_PretrainData/DATA_AFTERTRANS_SLICE"
    # maybe_mkdir_p(des_path)
    # for sub in listdir(ori_path):
    #     if not sub.endswith("npz"):
    #         continue
    #     sub_npy = np.load(join(ori_path, sub))['data'][0]
    #     print("handling ", sub)
    #     for i in range(sub_npy.shape[0]):
    #         save_npy = sub_npy[i]
    #         np.savez(join(des_path, sub[:-4] + "_" + str(i) + ".npz"), data=save_npy)
    # break
