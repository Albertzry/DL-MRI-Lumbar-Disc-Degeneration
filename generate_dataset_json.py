#!/usr/bin/env python3
"""
生成dataset.json的脚本
扫描指定目录并生成与nnFormer格式完全一致的dataset.json文件
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any


def scan_directory_structure(base_path: str) -> Dict[str, Any]:
    """
    扫描目录结构并生成dataset.json数据
    
    Args:
        base_path: 数据集根目录路径
        
    Returns:
        包含dataset.json数据的字典
    """
    base_path = Path(base_path)
    
    # 定义目录路径
    images_tr_dir = base_path / "imagesTr"
    labels_tr_dir = base_path / "labelsTr"
    images_ts_dir = base_path / "imagesTs"
    labels_ts_dir = base_path / "labelsTs"
    
    # 检查目录是否存在
    if not images_tr_dir.exists():
        raise FileNotFoundError(f"训练图像目录不存在: {images_tr_dir}")
    if not labels_tr_dir.exists():
        raise FileNotFoundError(f"训练标签目录不存在: {labels_tr_dir}")
    if not images_ts_dir.exists():
        raise FileNotFoundError(f"测试图像目录不存在: {images_ts_dir}")
    
    # 扫描训练数据
    training_data = []
    training_images = sorted(glob.glob(str(images_tr_dir / "*.nii.gz")))
    
    for img_path in training_images:
        img_name = os.path.basename(img_path)
        label_name = img_name  # 假设图像和标签文件名相同
        
        # 检查对应的标签文件是否存在
        label_path = labels_tr_dir / label_name
        if label_path.exists():
            training_data.append({
                "image": f"./imagesTr/{img_name}",
                "label": f"./labelsTr/{label_name}"
            })
        else:
            print(f"警告: 未找到对应的标签文件: {label_path}")
    
    # 扫描测试数据
    test_data = []
    test_images = sorted(glob.glob(str(images_ts_dir / "*.nii.gz")))
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        test_data.append(f"./imagesTs/{img_name}")
    
    # 构建dataset.json结构
    dataset_info = {
        "description": "Segmentation",
        "labels": {
            "0": "background",
            "1": "Segmentation_1",
            "2": "Segmentation_2"
        },
        "licence": "see challenge website",
        "modality": {
            "0": "MRI"
        },
        "name": "hospital_data",
        "numTraining": len(training_data),
        "numTest": len(test_data),
        "reference": "see challenge website",
        "release": "0.0",
        "tensorImageSize": "4D",
        "training": training_data,
        "test": test_data
    }
    
    return dataset_info


def generate_dataset_json(base_path: str, output_path: str = None) -> None:
    """
    生成dataset.json文件
    
    Args:
        base_path: 数据集根目录路径
        output_path: 输出文件路径，默认为base_path/dataset.json
    """
    try:
        # 扫描目录结构
        print(f"正在扫描目录: {base_path}")
        dataset_info = scan_directory_structure(base_path)
        
        # 确定输出路径
        if output_path is None:
            output_path = os.path.join(base_path, "dataset.json")
        
        # 写入JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)
        
        print(f"✅ dataset.json 已成功生成: {output_path}")
        print(f"📊 统计信息:")
        print(f"   - 训练样本数: {dataset_info['numTraining']}")
        print(f"   - 测试样本数: {dataset_info['numTest']}")
        
    except Exception as e:
        print(f"❌ 生成dataset.json时出错: {str(e)}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成nnFormer格式的dataset.json文件')
    parser.add_argument('--base_path', 
                       default='/root/DL-MRI-Lumbar-Disc-Degeneration/DATASET/nnFormer_raw/nnFormer_raw_data/Task01_disc',
                       help='数据集根目录路径')
    parser.add_argument('--output', 
                       help='输出文件路径（可选，默认为base_path/dataset.json）')
    
    args = parser.parse_args()
    
    # 检查基础路径是否存在
    if not os.path.exists(args.base_path):
        print(f"❌ 错误: 目录不存在: {args.base_path}")
        return
    
    # 生成dataset.json
    generate_dataset_json(args.base_path, args.output)


if __name__ == "__main__":
    main()
