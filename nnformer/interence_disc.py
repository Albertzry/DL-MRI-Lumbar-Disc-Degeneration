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
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def binarize(pred, threshold=0.5):
    try:
        pred_arr = pred.astype(np.float32)
    except Exception:
        pred_arr = np.array(pred, dtype=np.float32)
    # If prediction already integer labels (0/1), thresholding still works.
    return pred_arr > threshold

def dice(pred, label):
    pred_bool = pred.astype(bool)
    label_bool = label.astype(bool)
    if (pred_bool.sum() + label_bool.sum()) == 0:
        return 1.0
    else:
        return 2.0 * np.logical_and(pred_bool, label_bool).sum() / (pred_bool.sum() + label_bool.sum())

def hd95(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        try:
            return binary.hd95(pred_bool, gt_bool)
        except Exception as e:
            # fallback: return 0 on error but print warning
            print("Warning: hd95 calculation failed for a case:", e)
            return 0.0
    else:
        return 0.0

def process_label(label):

    return label > 0

def test(fold, threshold=0.5):
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
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fw = open(os.path.join(out_dir, 'dice_pre.txt'), 'w')

    # iterate by filename order; assumes matching order/name between labels and predictions
    for label_path, infer_path in zip(label_list, infer_list):
        name = os.path.basename(label_path)
        print('Processing:', name)
        fw.write('*' * 20 + '\\n')
        fw.write(name + '\\n')

        label = read_nii(label_path)
        pred = read_nii(infer_path)

        # convert labels and predictions to binary masks
        label_mask = process_label(label)
        pred_mask = binarize(pred, threshold=threshold)

        d = dice(pred_mask, label_mask)
        hd = hd95(pred_mask, label_mask)

        Dice_list.append(d)
        HD_list.append(hd)

        fw.write('Dice: {:.4f}\\n'.format(d))
        fw.write('HD95: {:.4f}\\n'.format(hd))

    # summary
    fw.write('*' * 20 + '\\n')
    fw.write('Mean_Dice: {:.4f}\\n'.format(np.mean(Dice_list) if len(Dice_list) > 0 else 0.0))
    fw.write('Mean_HD95: {:.4f}\\n'.format(np.mean(HD_list) if len(HD_list) > 0 else 0.0))
    fw.write('*' * 20 + '\\n')
    fw.close()

    print('Done. Results saved to', os.path.join(out_dir, 'dice_pre.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference evaluation for DISC/IVD dataset (nnFormer)')
    parser.add_argument('fold', help='fold name (subfolder inside ./inferTs containing predictions)')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold to binarize probabilistic predictions (default 0.5)')
    args = parser.parse_args()
    test(args.fold, threshold=args.threshold)
