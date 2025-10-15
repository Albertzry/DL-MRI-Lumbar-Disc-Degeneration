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

# 预处理统一到固定体素网格后再做中心裁剪
RESAMPLE_TARGET_SIZE = (512, 512, 85)  # ITK 顺序 (X, Y, Z)
CENTER_CROP_SIZE = (85, 256, 216)      # numpy 顺序 (Z, X, Y) -> 物理尺寸 [216, 256, 85]


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
    """利用 SimpleITK 将影像重采样到目标大小。"""
    target_size = tuple(int(i) for i in target_size)
    original_size = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    target_spacing = tuple(
        original_spacing[i] * (original_size[i] / float(target_size[i])) for i in range(len(target_size))
    )
    interpolator = sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear
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


def load_case_from_list_of_files(data_files, seg_file=None, target_size=RESAMPLE_TARGET_SIZE):
    """读取一例病例，按需重采样到统一体素网格。"""
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

    if target_size is not None:
        data_itk = [resample_sitk_image_to_size(d, target_size, is_seg=False) for d in data_itk]
        if seg_itk is not None:
            seg_itk = resample_sitk_image_to_size(seg_itk, target_size, is_seg=True)
        properties["size_after_resample"] = np.array(target_size)[[2, 1, 0]]
        properties["spacing_after_resample"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    else:
        properties["size_after_resample"] = properties["original_size_of_raw_data"]
        properties["spacing_after_resample"] = properties["original_spacing"]

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk]).astype(np.float32)
    if seg_itk is not None:
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
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