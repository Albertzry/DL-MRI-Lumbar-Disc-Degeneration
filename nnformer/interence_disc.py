import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary

"""
inference_disc.py
Created to match the style of inference_acdc.py / inference_synapse.py / inference_tumor.py
and adapted for the IVD / DISC binary task described in the provided thesis (nnFormer setup).

Behavior:
- expects ground-truth NIfTI files in ./labelsTs/*.nii.gz
- expects predicted NIfTI files in ./inferTs/<fold>/*.nii.gz
- writes a dice_pre.txt in ./inferTs/<fold>/ with per-case Dice and HD95 and final averages
- supports thresholding of probabilistic outputs (default 0.5)
"""

def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = itk_img.GetSpacing()
    arr = sitk.GetArrayFromImage(itk_img)
    return arr, spacing

def binarize(pred, threshold=0.5):
    try:
        pred_arr = pred.astype(np.float32)
    except Exception:
        pred_arr = np.array(pred, dtype=np.float32)
    # If prediction already integer labels (0/1), thresholding still works.
    return pred_arr > threshold

def dice(pred, label, soft=False, eps=1e-6, empty_policy='one'):
    """
    Compute Dice. If soft=True, treat `pred` as probabilities.
    empty_policy: 'one' -> when both empty return 1.0 (default),
                  'zero' -> return 0.0,
                  'exclude' -> return None (caller should skip this case).
    """
    pred_f = np.array(pred, dtype=np.float32)
    label_f = np.array(label, dtype=np.float32)

    if not soft:
        pred_f = (pred_f > 0.5).astype(np.float32)
        label_f = (label_f > 0.5).astype(np.float32)

    inter = np.sum(pred_f * label_f)
    den = np.sum(pred_f) + np.sum(label_f)

    if den == 0:
        if empty_policy == 'one':
            return 1.0
        elif empty_policy == 'zero':
            return 0.0
        else:
            return None
    return (2.0 * inter + eps) / (den + eps)

def hd95(pred, gt, spacing=None, hd_in_mm=False):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        val = binary.hd95(pred_bool, gt_bool)
        # medpy returns distance in voxels; if dataset is isotropic and hd_in_mm=True,
        # convert voxels->mm by multiplying with spacing (use mean spacing as approximation)
        if hd_in_mm and spacing is not None:
            return float(val * float(np.mean(spacing)))
        return float(val)
    else:
        return 0.0

def process_label(label):
    return label > 0

def test(fold, threshold=0.5, soft=False, eps=1e-6, empty_policy='one', aggregate='case', hd_in_mm=False):
    path = './'
    label_list = sorted(glob.glob(os.path.join(path, 'labelsTs', '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(path, 'inferTs', fold, '*nii.gz')))

    if len(label_list) == 0:
        print('No ground-truth files found in ./labelsTs.')
        return
    if len(infer_list) == 0:
        print(f'No inference files found in ./inferTs/{fold}.')
        return

    print('Found', len(label_list), 'labels and', len(infer_list), 'predictions.')
    Dice_list = []
    HD_list = []

    out_dir = os.path.join(path, 'inferTs', fold)
    os.makedirs(out_dir, exist_ok=True)
    fw = open(os.path.join(out_dir, 'dice_pre.txt'), 'w')

    # map predictions by basename to avoid ordering mismatch
    infer_map = {os.path.basename(p): p for p in infer_list}

    # for global aggregation (voxel-wise pooled dice)
    global_num = 0.0
    global_den = 0.0

    for label_path in label_list:
        name = os.path.basename(label_path)
        print('Processing:', name)
        fw.write('*' * 20 + '\n')
        fw.write(name + '\n')

        if name not in infer_map:
            print(f'Warning: no matching prediction for {name} in inferTs/{fold}, skipping')
            fw.write(f'Warning: no matching prediction for {name}\n')
            continue

        infer_path = infer_map[name]

        label_arr, spacing = read_nii(label_path)
        pred_arr, _ = read_nii(infer_path)

        # convert labels and predictions to binary masks (or keep pred as prob when soft=True)
        label_mask = process_label(label_arr)
        if soft:
            pred_mask = np.array(pred_arr, dtype=np.float32)
        else:
            pred_mask = binarize(pred_arr, threshold=threshold)

        d = dice(pred_mask, label_mask, soft=soft, eps=eps, empty_policy=empty_policy)
        if d is None:
            # case excluded by policy
            fw.write('Dice: excluded (empty)\n')
        else:
            Dice_list.append(d)
            fw.write('Dice: {:.6f}\n'.format(d))

        # hd95 uses binary masks; when pred is soft we threshold at 0.5 for surface metrics
        hd = hd95((pred_mask > 0.5) if soft else pred_mask, label_mask, spacing=spacing, hd_in_mm=hd_in_mm)
        HD_list.append(hd)
        fw.write('HD95: {:.6f}\n'.format(hd))

        # accumulate for pooled/global Dice if requested
        if aggregate == 'global':
            p = (pred_mask > 0.5).astype(np.float32) if not soft else pred_mask.astype(np.float32)
            g = label_mask.astype(np.float32)
            global_num += 2.0 * np.sum(p * g)
            global_den += np.sum(p) + np.sum(g)

    # summary
    fw.write('*' * 20 + '\n')
    if aggregate == 'case':
        mean_dice = np.mean(Dice_list) if len(Dice_list) > 0 else 0.0
        fw.write('Mean_Dice: {:.6f}\n'.format(mean_dice))
    else:
        pooled = (global_num + eps) / (global_den + eps) if global_den > 0 else 0.0
        fw.write('Pooled_Dice: {:.6f}\n'.format(pooled))

    fw.write('Mean_HD95: {:.6f}\n'.format(np.mean(HD_list) if len(HD_list) > 0 else 0.0))
    fw.write('*' * 20 + '\n')
    fw.close()

    print('Done. Results saved to', os.path.join(out_dir, 'dice_pre.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference evaluation for DISC/IVD dataset (nnFormer)')
    parser.add_argument('fold', help='fold name (subfolder inside ./inferTs containing predictions)')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold to binarize probabilistic predictions (default 0.5)')
    parser.add_argument('--soft', action='store_true', help='treat predictions as probabilities and compute soft Dice when specified')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon for numerical stability in Dice')
    parser.add_argument('--empty-policy', choices=['one','zero','exclude'], default='one', help='how to treat cases where both pred and gt are empty')
    parser.add_argument('--aggregate', choices=['case','global'], default='case', help='aggregate Dice per case (case) or pooled/global over voxels (global)')
    parser.add_argument('--hd-mm', action='store_true', help='attempt to convert hd95 from voxels to mm using image spacing (assumes isotropic spacing)')
    args = parser.parse_args()
    test(args.fold, threshold=args.threshold, soft=args.soft, eps=args.eps, empty_policy=args.empty_policy, aggregate=args.aggregate, hd_in_mm=args.hd_mm)
