#!/usr/bin/env python3
"""
不转换 .npz，直接计算 Dice/HD95
"""
import os, glob, tempfile, shutil, numpy as np, nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnformer.evaluation.evaluator import evaluate_folder


def eval_npz_direct(gt_dir: str, npz_dir: str, labels: tuple, tmp_prefix="evalnpz"):
    """
    把 .npz 现场 argmax→nii，调用官方 evaluate_folder，结果不落地。
    """
    tmp_pred = tempfile.mkdtemp(prefix=tmp_prefix)
    try:
        for npz_file in glob.glob(os.path.join(npz_dir, "*.npz")):
            base   = os.path.basename(npz_file)[:-4]
            prob   = np.load(npz_file)['softmax']          # (C, Z, Y, X)
            seg    = prob.argmax(0).astype(np.uint8)       # (Z, Y, X)

            # 读属性
            pkl      = npz_file[:-4] + ".pkl"
            props    = load_pickle(pkl)
            ori_shape = props['original_size_of_raw_data']
            bbox      = props['crop_bbox']

            # 贴回原始几何
            ori_seg = np.zeros(ori_shape, dtype=np.uint8)
            ori_seg[bbox[0][0]:bbox[0][1],
                    bbox[1][0]:bbox[1][1],
                    bbox[2][0]:bbox[2][1]] = seg

            # 保存临时 nii
            out_nii = os.path.join(tmp_pred, base + ".nii.gz")
            img = nib.Nifti1Image(ori_seg, affine=props.get('nifti_affine', np.eye(4)))
            nib.save(img, out_nii)

        # 官方评估
        return evaluate_folder(gt_dir, tmp_pred, labels, compute_assd=True)
    finally:
        shutil.rmtree(tmp_pred, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser("Evaluate .npz softmax directly")
    parser.add_argument('-ref', required=True, help="ground-truth labels folder (nii.gz)")
    parser.add_argument('-pred', required=True, help="predicted .npz folder")
    parser.add_argument('-l', nargs='+', type=int, required=True, help="label IDs to evaluate, e.g. 1 2")
    args = parser.parse_args()

    res = eval_npz_direct(args.ref, args.pred, tuple(args.l))
    print("\n=== Global Average ===")
    for l in args.l:
        print(f"label {l}  |  "
              f"Dice: {res['mean']['Dice'][l]:.4f} ± {res['std']['Dice'][l]:.4f}  |  "
              f"HD95: {res['mean']['HD95'][l]:.2f} ± {res['std']['HD95'][l]:.2f} mm")


if __name__ == "__main__":
    main()