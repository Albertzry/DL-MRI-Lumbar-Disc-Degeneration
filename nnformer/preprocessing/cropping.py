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

import os
import pickle

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndi_rotate, gaussian_filter as ndi_gaussian_filter, zoom as ndi_zoom

# 预处理统一到固定体素网格后再做中心裁剪
RESAMPLE_TARGET_SIZE = (512, 512, 85)  # ITK 顺序 (X, Y, Z)
CENTER_CROP_SIZE = (85, 256, 216)      # numpy 顺序 (Z, X, Y) -> 物理尺寸 [216, 256, 85]

# 在重采样后启用数据增强
AUGMENT_AT_RESAMPLE = True

# 增强超参数（概率与强度范围）
AUG_CFG = {
    "p_rotate": 0.5,
    "max_rotate_deg": 3.0,  # 每个轴的最大旋转角度
    "p_flip": 0.5,           # 针对每个轴独立判定是否翻转
    "allow_flip_z": False,   # 是否允许 Z 轴翻转（按需关闭）
    "p_gamma": 0.3,
    "gamma_range": (0.9, 1.1),
    "p_brightness_contrast": 0.3,
    "contrast_range": (0.9, 1.1),
    "brightness_std_rel": 0.1,  # 亮度偏移的相对标准差（相对每通道 std）
    "p_noise": 0.3,
    "noise_std_rel": (0.01, 0.05),
    "p_blur": 0.3,
    "blur_sigma_range": (0.5, 1.2),  # 以体素为单位
    "p_lowres": 0.3,
    "lowres_factor_range": (0.5, 0.8),  # 缩小因子，随后再放回原尺寸
    "seed": None,
}


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def resample_sitk_image_to_size(itk_image, target_size, is_seg=False):
    """保留：使用 SimpleITK 重采样（未使用）。保留以兼容旧流程。"""
    target_size = tuple(int(i) for i in target_size)
    original_size = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    target_spacing = tuple(
        original_spacing[i] * (original_size[i] / float(target_size[i])) for i in range(len(target_size))
    )
    interpolator = sitk.sitkLinear
    return sitk.Resample(
        itk_image,
        target_size,
        sitk.Transform(),
        interpolator,
        itk_image.GetOrigin(),
        target_spacing,
        itk_image.GetDirection(),
        0,
        itk_image.GetPixelID(),
    )


def _torch_interpolate_3d(arr_czyx, target_size_xyz):
    """使用 PyTorch 对 3D 体数据做三线性插值。

    入参:
    - arr_czyx: numpy 数组，形状 (C, Z, Y, X)
    - target_size_xyz: 目标体素数，顺序 (X, Y, Z)

    返回:
    - 同 dtype 的 numpy 数组，形状 (C, Z, Y, X)
    """
    assert arr_czyx.ndim == 4, "expected (C, Z, Y, X)"
    tx, ty, tz = [int(i) for i in target_size_xyz]
    # torch 需要 (N, C, D, H, W)，size=(D, H, W)=(Z, Y, X)
    t = torch.from_numpy(arr_czyx).unsqueeze(0).float()
    out = F.interpolate(t, size=(tz, ty, tx), mode="trilinear", align_corners=False)
    return out.squeeze(0).cpu().numpy()


def _rand(rng, low, high):
    return float(rng.uniform(low, high))


def _maybe(p, rng):
    return rng.random() < p


def _apply_intensity_ops(data_czyx, rng, cfg):
    """仅对图像执行强度类增强，保持形状不变。
    data_czyx: (C, Z, Y, X) float32
    返回：增强后的 data_czyx
    """
    C = data_czyx.shape[0]
    out = data_czyx.copy()

    # Gamma 校正（对每通道单独归一化到 0-1 再做 gamma）
    if _maybe(cfg["p_gamma"], rng):
        gamma = _rand(rng, *cfg["gamma_range"])
        eps = 1e-8
        for c in range(C):
            ch = out[c]
            vmin, vmax = float(ch.min()), float(ch.max())
            if vmax - vmin > eps:
                ch_norm = (ch - vmin) / (vmax - vmin + eps)
                ch_aug = np.power(ch_norm, gamma)
                out[c] = ch_aug * (vmax - vmin) + vmin

    # 亮度/对比度
    if _maybe(cfg["p_brightness_contrast"], rng):
        alpha = _rand(rng, *cfg["contrast_range"])  # 对比度
        for c in range(C):
            ch = out[c]
            std = float(ch.std())
            beta = rng.normal(0.0, cfg["brightness_std_rel"] * (std + 1e-8))  # 亮度偏移
            out[c] = ch * alpha + beta

    # 高斯噪声
    if _maybe(cfg["p_noise"], rng):
        std_low, std_high = cfg["noise_std_rel"]
        for c in range(C):
            ch = out[c]
            ch_std = float(ch.std())
            noise_std = _rand(rng, std_low, std_high) * (ch_std + 1e-8)
            out[c] = ch + rng.normal(0.0, noise_std, size=ch.shape).astype(np.float32)

    # 模糊（3D 高斯滤波）
    if _maybe(cfg["p_blur"], rng):
        sigma = _rand(rng, *cfg["blur_sigma_range"])  # 各向同性
        for c in range(C):
            out[c] = ndi_gaussian_filter(out[c], sigma=sigma)

    # 低分辨率模拟：下采样再上采样回原尺寸
    if _maybe(cfg["p_lowres"], rng):
        # 仅在 XY 平面降采样/上采样，保持 Z 不变
        factor = _rand(rng, *cfg["lowres_factor_range"])  # 0.5~0.8
        _, Z, Y, X = out.shape
        new_Z = Z
        new_Y = max(1, int(round(Y * factor)))
        new_X = max(1, int(round(X * factor)))
        for c in range(C):
            low = ndi_zoom(out[c], (1.0, new_Y / Y, new_X / X), order=1)
            out[c] = ndi_zoom(low, (1.0, Y / low.shape[1], X / low.shape[2]), order=1)

    return out


def _apply_geometric_ops(data_czyx, seg_zyx, rng, cfg):
    """几何增强：作用于图像与标签。保持输出形状不变。
    data_czyx: (C, Z, Y, X)
    seg_zyx: (Z, Y, X) 或 None
    返回：data_czyx_aug, seg_zyx_aug
    """
    C, Z, Y, X = data_czyx.shape
    d_out = data_czyx.copy()
    s_out = None if seg_zyx is None else seg_zyx.copy()

    # 随机翻转（每轴独立判定）
    if _maybe(cfg["p_flip"], rng):
        # 可选：Z 轴翻转，按需关闭
        if cfg.get("allow_flip_z", False) and _maybe(0.5, rng):  # Z 轴
            d_out = d_out[:, ::-1, :, :]
            if s_out is not None:
                s_out = s_out[::-1, :, :]
        if _maybe(0.5, rng):  # Y 轴
            d_out = d_out[:, :, ::-1, :]
            if s_out is not None:
                s_out = s_out[:, ::-1, :]
        if _maybe(0.5, rng):  # X 轴
            d_out = d_out[:, :, :, ::-1]
            if s_out is not None:
                s_out = s_out[:, :, ::-1]

    # 随机小角度旋转（绕三个轴，保持形状）
    if _maybe(cfg["p_rotate"], rng):
        max_deg = cfg["max_rotate_deg"]
        # 依次对三个平面旋转：
        deg_z = _rand(rng, -max_deg, max_deg)  # 绕 Z 轴，相当于在 (Y, X) 平面旋转
        deg_y = _rand(rng, -max_deg, max_deg)  # 绕 Y 轴 -> (Z, X)
        deg_x = _rand(rng, -max_deg, max_deg)  # 绕 X 轴 -> (Z, Y)
        # 对每个通道分别处理
        for c in range(C):
            ch = d_out[c]
            if abs(deg_z) > 1e-3:
                ch = ndi_rotate(ch, angle=deg_z, axes=(1, 2), reshape=False, order=1, mode='nearest')
            if abs(deg_y) > 1e-3:
                ch = ndi_rotate(ch, angle=deg_y, axes=(0, 2), reshape=False, order=1, mode='nearest')
            if abs(deg_x) > 1e-3:
                ch = ndi_rotate(ch, angle=deg_x, axes=(0, 1), reshape=False, order=1, mode='nearest')
            d_out[c] = ch
        if s_out is not None:
            s = s_out
            if abs(deg_z) > 1e-3:
                s = ndi_rotate(s, angle=deg_z, axes=(1, 2), reshape=False, order=0, mode='nearest')
            if abs(deg_y) > 1e-3:
                s = ndi_rotate(s, angle=deg_y, axes=(0, 2), reshape=False, order=0, mode='nearest')
            if abs(deg_x) > 1e-3:
                s = ndi_rotate(s, angle=deg_x, axes=(0, 1), reshape=False, order=0, mode='nearest')
            s_out = s.astype(np.int16, copy=False)

    return d_out, s_out


def apply_augmentations(data_czyx, seg_zyx=None, cfg=AUG_CFG):
    """在重采样后应用增强。几何增强同步作用到 seg，强度增强仅作用于 data。
    返回：data_aug, seg_aug
    """
    rng = np.random.default_rng(cfg.get("seed", None))
    # 先几何，再强度（强度需在最终对齐后的图像上进行）
    data_geo, seg_geo = _apply_geometric_ops(data_czyx, seg_zyx, rng, cfg)
    data_out = _apply_intensity_ops(data_geo, rng, cfg)
    return data_out, seg_geo


def center_crop_to_size(data, seg=None, target_size=CENTER_CROP_SIZE, nonzero_label=-1):
    """按中心裁剪/填充到目标尺寸，保持图像与标签对齐。"""
    assert len(data.shape) == 4, "data must have shape (C, Z, X, Y)"
    _, orig_z, orig_x, orig_y = data.shape
    target_z, target_x, target_y = [int(i) for i in target_size]

    def compute_crop_and_pad(original, target):
        if original >= target:
            start = (original - target) // 2
            end = start + target
            pad_before, pad_after = 0, 0
        else:
            start, end = 0, original
            total_pad = target - original
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        return start, end, pad_before, pad_after

    z_start, z_end, z_pad_before, z_pad_after = compute_crop_and_pad(orig_z, target_z)
    x_start, x_end, x_pad_before, x_pad_after = compute_crop_and_pad(orig_x, target_x)
    y_start, y_end, y_pad_before, y_pad_after = compute_crop_and_pad(orig_y, target_y)

    cropped = data[:, z_start:z_end, x_start:x_end, y_start:y_end]
    if seg is not None:
        cropped_seg = seg[:, z_start:z_end, x_start:x_end, y_start:y_end]
    else:
        cropped_seg = None

    if any([z_pad_before, z_pad_after, x_pad_before, x_pad_after, y_pad_before, y_pad_after]):
        pad_config = (
            (0, 0),
            (z_pad_before, z_pad_after),
            (x_pad_before, x_pad_after),
            (y_pad_before, y_pad_after),
        )
        cropped = np.pad(cropped, pad_config, mode="constant", constant_values=0)
        if cropped_seg is not None:
            cropped_seg = np.pad(cropped_seg, pad_config, mode="constant", constant_values=nonzero_label)

    if cropped.shape[1:] != (target_z, target_x, target_y):
        cropped = cropped[:, :target_z, :target_x, :target_y]
        if cropped_seg is not None:
            cropped_seg = cropped_seg[:, :target_z, :target_x, :target_y]

    if cropped_seg is None:
        nz_mask = np.any(cropped != 0, axis=0)
        seg_mask = np.full((1, target_z, target_x, target_y), nonzero_label, dtype=np.int16)
        seg_mask[0][nz_mask] = 0
        cropped_seg = seg_mask
    else:
        cropped_seg = cropped_seg.astype(np.int16, copy=False)

    bbox = [[z_start, z_end], [x_start, x_end], [y_start, y_end]]
    return cropped.astype(np.float32, copy=False), cropped_seg.astype(np.float32, copy=False), bbox


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None, target_size=RESAMPLE_TARGET_SIZE, apply_aug=AUGMENT_AT_RESAMPLE):
    """读取一例病例，使用 PyTorch 对图像与标签做 3D 三线性插值到统一体素网格，
    并将标签在插值后映射回离散集合 {0,1,2}。"""
    assert isinstance(data_files, (list, tuple)), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]
    seg_itk = sitk.ReadImage(seg_file) if seg_file is not None else None

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    # 先转为 numpy (Z, Y, X)，再堆成 (C, Z, Y, X)
    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk]).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg_itk) if seg_itk is not None else None

    if target_size is not None:
        # 使用 PyTorch 做三线性插值，输出数组 (C, Z, Y, X)
        data_npy = _torch_interpolate_3d(data_npy, target_size).astype(np.float32)
        if seg_arr is not None:
            seg_float = seg_arr.astype(np.float32, copy=False)
            seg_res = _torch_interpolate_3d(seg_float[None], target_size)[0]
            # 离散化回 {0,1,2}
            seg_res = np.rint(seg_res)
            seg_res = np.clip(seg_res, 0, 2).astype(np.int16, copy=False)
            seg_npy = seg_res[None].astype(np.float32)
        else:
            seg_npy = None

        # 计算 spacing_after_resample（基于原 spacing/size 推导）
        orig_size_xyz = np.array(data_itk[0].GetSize())  # (X, Y, Z)
        orig_spacing_xyz = np.array(data_itk[0].GetSpacing())  # (X, Y, Z)
        tgt_size_xyz = np.array([int(i) for i in target_size])
        tgt_spacing_xyz = orig_spacing_xyz * (orig_size_xyz / tgt_size_xyz.astype(np.float32))
        properties["size_after_resample"] = tgt_size_xyz[[2, 1, 0]]  # (Z, Y, X)
        properties["spacing_after_resample"] = tgt_spacing_xyz[[2, 1, 0]]  # (Z, Y, X)
    else:
        # 不重采样
        if seg_arr is not None:
            seg_npy = seg_arr[None].astype(np.float32)
        else:
            seg_npy = None
        properties["size_after_resample"] = properties["original_size_of_raw_data"]
        properties["spacing_after_resample"] = properties["original_spacing"]

    # 在重采样完成后应用数据增强
    if apply_aug:
        seg_zyx = None if seg_npy is None else seg_npy[0].astype(np.int16)
        data_npy, seg_aug = apply_augmentations(data_npy, seg_zyx, AUG_CFG)
        if seg_aug is not None:
            # 保障标签仍为 {0,1,2} 集合
            seg_aug = np.clip(np.rint(seg_aug), 0, 2).astype(np.int16, copy=False)
            seg_npy = seg_aug[None].astype(np.float32)
        else:
            seg_npy = None
    return data_npy, seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None, target_size=RESAMPLE_TARGET_SIZE):
        """
        按需重采样 + 中心裁剪，保持与原有调用兼容。
        :param num_threads: 预处理使用的线程数
        :param output_folder: 结果输出目录
        :param target_size: 重采样目标体素数 (X, Y, Z)
        """
        self.output_folder = output_folder
        self.num_threads = num_threads
        self.target_size = target_size

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = center_crop_to_size(data, seg, target_size=CENTER_CROP_SIZE, nonzero_label=-1)
        shape_after = data.shape
        spacing_to_show = properties.get("spacing_after_resample", properties.get("original_spacing"))
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(spacing_to_show), "\n")

        properties["crop_bbox"] = bbox
        if seg is not None:
            properties['classes'] = np.unique(seg)
            seg[seg < -1] = 0
        else:
            properties['classes'] = None
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None, target_size=RESAMPLE_TARGET_SIZE):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file, target_size=target_size)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(
                    case[:-1],
                    case[-1],
                    target_size=self.target_size,
                )

                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)