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
import shutil
import numpy as np
import SimpleITK as sitk

from multiprocessing import Pool
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p,
    subfiles,
)

# 先重采样到固定体素网格，再居中裁剪到指定输出尺寸
RESAMPLE_TARGET_SIZE = (512, 512, 85)  # ITK 顺序 (X, Y, Z)
CENTER_CROP_SIZE = (85, 256, 216)      # numpy 顺序 (Z, Y, X)，对应物理尺寸 [216, 256, 85]


def create_nonzero_mask(data):
    """生成非零掩膜，用于限定实际成像区域。"""
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, Z, Y, X) or (Z, Y, X)"
    nonzero_mask = np.zeros(data.shape[-3:], dtype=bool)
    if data.ndim == 4:
        for c in range(data.shape[0]):
            nonzero_mask |= data[c] != 0
    else:
        nonzero_mask |= data != 0
    return binary_fill_holes(nonzero_mask)


def get_bbox_from_mask(mask, outside_value=0):
    coords = np.where(mask != outside_value)
    if coords[0].size == 0:
        return [[0, mask.shape[0]], [0, mask.shape[1]], [0, mask.shape[2]]]
    zmin, zmax = coords[0].min(), coords[0].max() + 1
    ymin, ymax = coords[1].min(), coords[1].max() + 1
    xmin, xmax = coords[2].min(), coords[2].max() + 1
    return [[zmin, zmax], [ymin, ymax], [xmin, xmax]]


def crop_to_bbox(data, bbox):
    slices = tuple(slice(b[0], b[1]) for b in bbox)
    if data.ndim == 4:
        return data[:, slices[0], slices[1], slices[2]]
    return data[slices[0], slices[1], slices[2]]


def resample_sitk_image_to_size(itk_image, target_size, is_seg=False):
    """利用 SimpleITK 将影像重采样到目标体素尺寸。"""
    orig_size = itk_image.GetSize()
    orig_spacing = itk_image.GetSpacing()
    target_size = tuple(int(i) for i in target_size)
    target_spacing = tuple(
        orig_spacing[i] * (orig_size[i] / float(target_size[i])) for i in range(len(orig_size))
    )
    interp = sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear
    return sitk.Resample(
        itk_image,
        target_size,
        sitk.Transform(),
        interp,
        itk_image.GetOrigin(),
        target_spacing,
        itk_image.GetDirection(),
        0,
        itk_image.GetPixelID(),
    )


def get_case_identifier(case):
    return case[0].split("/")[-1].split(".nii.gz")[0][:-5]


def get_case_identifier_from_npz(case):
    return case.split("/")[-1][:-4]


def load_case_from_list_of_files(data_files, seg_file=None, target_size=RESAMPLE_TARGET_SIZE):
    """读取一组模态 NIfTI 文件，按需重采样，再转成 numpy 数组。"""
    assert isinstance(data_files, (list, tuple)), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]
    seg_itk = sitk.ReadImage(seg_file) if seg_file is not None else None

    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file
    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
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

    data_npy = np.stack([sitk.GetArrayFromImage(img) for img in data_itk], axis=0).astype(np.float32)
    seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.int16) if seg_itk is not None else None

    return data_npy, seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """按非零体素紧凑裁剪，可快速剔除大区域背景。"""
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = data[
        :,
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1],
    ]

    if seg is not None:
        cropped_seg = seg[
            :,
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1],
        ]
    else:
        nz_mask = nonzero_mask[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1],
        ].astype(np.int16)
        cropped_seg = np.full((1,) + nz_mask.shape, nonzero_label, dtype=np.int16)
        cropped_seg[0][nz_mask > 0] = 0

    return cropped_data, cropped_seg, bbox


def center_crop_to_size(data, seg=None, target_size=CENTER_CROP_SIZE, nonzero_label=-1):
    """将数据居中裁剪到 target_size，不足时使用常数填充。"""
    assert len(data.shape) == 4, "data must have shape (C, Z, Y, X)"
    C, Z, Y, X = data.shape
    tz, ty, tx = map(int, target_size)

    z0 = (Z - tz) // 2
    y0 = (Y - ty) // 2
    x0 = (X - tx) // 2
    z1 = z0 + tz
    y1 = y0 + ty
    x1 = x0 + tx

    z0c, z1c = max(z0, 0), min(z1, Z)
    y0c, y1c = max(y0, 0), min(y1, Y)
    x0c, x1c = max(x0, 0), min(x1, X)

    bbox = [[int(z0c), int(z1c)], [int(y0c), int(y1c)], [int(x0c), int(x1c)]]

    cropped = data[:, z0c:z1c, y0c:y1c, x0c:x1c]
    if seg is not None:
        cropped_seg = seg[:, z0c:z1c, y0c:y1c, x0c:x1c]
    else:
        cropped_seg = None

    pad_z_before = max(0, -z0)
    pad_y_before = max(0, -y0)
    pad_x_before = max(0, -x0)
    pad_z_after = max(0, z1 - Z)
    pad_y_after = max(0, y1 - Y)
    pad_x_after = max(0, x1 - X)

    if any([pad_z_before, pad_z_after, pad_y_before, pad_y_after, pad_x_before, pad_x_after]):
        pad_width = (
            (0, 0),
            (pad_z_before, pad_z_after),
            (pad_y_before, pad_y_after),
            (pad_x_before, pad_x_after),
        )
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=0)
        if cropped_seg is not None:
            seg_pad_width = (
                (0, 0),
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            )
            cropped_seg = np.pad(cropped_seg, seg_pad_width, mode="constant", constant_values=nonzero_label)
        else:
            seg_pad_width = (
                (0, 0),
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            )

    if cropped.shape[1:] != (tz, ty, tx):
        cropped = cropped[:, :tz, :ty, :tx]
        if cropped_seg is not None:
            cropped_seg = cropped_seg[:, :tz, :ty, :tx]

    if cropped_seg is None:
        nz = np.any(cropped != 0, axis=0)
        nz_mask = np.full((tz, ty, tx), nonzero_label, dtype=np.int16)
        nz_mask[nz] = 0
        cropped_seg = nz_mask[None]

    return cropped, cropped_seg, bbox


class ImageCropper(object):
    """封装批量重采样 + 裁剪逻辑，支持多线程预处理。"""

    def __init__(self, num_threads, output_folder=None, target_size=RESAMPLE_TARGET_SIZE):
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
        properties["crop_bbox"] = bbox
        properties["data_shape_before_crop"] = shape_before
        properties["data_shape_after_crop"] = shape_after
        print(
            "before crop:",
            shape_before,
            "after crop:",
            shape_after,
            "spacing:",
            np.array(properties["spacing_after_resample"]) if "spacing_after_resample" in properties else np.array(properties["original_spacing"]),
            "\n",
        )
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        """直接从文件路径读取、重采样并裁剪，返回 numpy 数据。"""
        data, seg, properties = load_case_from_list_of_files(
            data_files,
            seg_file,
            target_size=RESAMPLE_TARGET_SIZE,
        )
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, f"{case_identifier}.npz"))
                        or not os.path.isfile(os.path.join(self.output_folder, f"{case_identifier}.pkl"))):

                data, seg, properties = load_case_from_list_of_files(
                    case[:-1],
                    case[-1],
                    target_size=self.target_size,
                )

                data, seg, properties = self.crop(data, properties, seg)

                all_data = np.concatenate((data, seg), axis=0)
                np.savez_compressed(os.path.join(self.output_folder, f"{case_identifier}.npz"), data=all_data)
                with open(os.path.join(self.output_folder, f"{case_identifier}.pkl"), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [os.path.basename(i)[:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for case in list_of_files:
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for case in list_of_files:
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, f"{case_identifier}.pkl"), 'rb') as f:
            return pickle.load(f)

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, f"{case_identifier}.pkl"), 'wb') as f:
            pickle.dump(properties, f)