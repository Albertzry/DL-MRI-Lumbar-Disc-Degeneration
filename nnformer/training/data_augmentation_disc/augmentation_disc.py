"""
数据增强模块 - 用于腰椎间盘退变的MRI图像
Data Augmentation for Lumbar Disc Degeneration MRI

支持的增强技术：
- 随机旋转 (Random 3D Rotation)
- 镜像翻转 (Mirror Flipping)
- Gamma校正 (Gamma Correction)
- 亮度调整 (Brightness Adjustment)
- 对比度调整 (Contrast Adjustment)
- 高斯噪声 (Gaussian Noise)
- 高斯模糊 (Gaussian Blur)
- 低分辨率模拟 (Low Resolution Simulation)

增强概率随epoch线性衰减
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, gaussian_filter


def torch_interpolate_3d_disc(data, target_size, mode='trilinear', is_seg=False):
    """
    使用PyTorch进行3D插值（用于低分辨率增强）
    
    :param data: numpy array with shape (C, Z, X, Y)
    :param target_size: tuple (Z, X, Y)
    :param mode: interpolation mode ('trilinear' for images, 'nearest' for segmentation)
    :param is_seg: whether this is segmentation data
    :return: interpolated numpy array
    """
    if is_seg:
        mode = 'nearest'
    
    # Convert to torch tensor
    data_torch = torch.from_numpy(data).unsqueeze(0).float()  # (1, C, Z, X, Y)
    
    # Interpolate
    target_size_tuple = tuple(int(i) for i in target_size)
    interpolated = F.interpolate(
        data_torch, 
        size=target_size_tuple, 
        mode=mode,
        align_corners=False if mode == 'trilinear' else None
    )
    
    # Convert back to numpy
    result = interpolated.squeeze(0).numpy()
    
    if is_seg:
        result = np.round(result).astype(np.int16)
    
    return result


class DataAugmentation3D_disc:
    """
    3D数据增强类，专门为腰椎间盘退变MRI设计
    
    特点：
    1. 保守的增强参数，保持脊柱解剖结构的有效性
    2. 图像和标签同步增强，保持空间一致性
    3. 增强概率随epoch线性衰减
    
    Example usage:
        # 创建增强器
        augmentor = DataAugmentation3D_disc(
            initial_p=0.2,      # 初始概率20%
            final_p=0.05,       # 最终概率5%
            max_epochs=1000     # 最大epoch数
        )
        
        # 在训练循环中
        for epoch in range(max_epochs):
            for data, seg in dataloader:
                # 应用增强（自动根据epoch调整概率）
                data_aug, seg_aug = augmentor(data, seg, current_epoch=epoch)
                # ... 训练步骤
    """
    
    def __init__(self, 
                 # 增强开关
                 do_rotation=True, 
                 do_mirror=True, 
                 do_gamma=True, 
                 do_brightness=True, 
                 do_contrast=True, 
                 do_noise=True, 
                 do_blur=True, 
                 do_low_res=True,
                 # 增强参数（针对腰椎间盘优化）
                 rotation_angle_range=(-15, 15),
                 mirror_axes=(0, 1, 2),
                 gamma_range=(0.7, 1.5),
                 brightness_range=(-0.2, 0.2),
                 contrast_range=(0.75, 1.25),
                 noise_variance=(0, 0.05),
                 blur_sigma=(0.5, 1.5),
                 low_res_scale=(0.5, 1.0),
                 # 概率衰减参数
                 initial_p=0.2,      # 初始概率
                 final_p=0.05,       # 最终概率
                 max_epochs=1000):   # 最大epoch数
        """
        初始化腰椎间盘MRI数据增强器
        
        参数说明：
        :param do_rotation: 是否应用旋转增强
        :param rotation_angle_range: 旋转角度范围（度），保守设置±15°
        :param do_mirror: 是否应用镜像翻转
        :param mirror_axes: 可镜像的轴 (0=Z, 1=X, 2=Y)
        :param do_gamma: 是否应用gamma校正
        :param gamma_range: gamma值范围
        :param do_brightness: 是否调整亮度
        :param brightness_range: 亮度调整范围
        :param do_contrast: 是否调整对比度
        :param contrast_range: 对比度乘数范围
        :param do_noise: 是否添加高斯噪声
        :param noise_variance: 噪声方差范围
        :param do_blur: 是否应用高斯模糊
        :param blur_sigma: 模糊sigma范围
        :param do_low_res: 是否模拟低分辨率
        :param low_res_scale: 下采样比例范围
        :param initial_p: 初始增强概率（epoch 0时）
        :param final_p: 最终增强概率（max_epochs时）
        :param max_epochs: 最大训练epoch数，用于计算衰减
        """
        # 增强开关
        self.do_rotation = do_rotation
        self.do_mirror = do_mirror
        self.do_gamma = do_gamma
        self.do_brightness = do_brightness
        self.do_contrast = do_contrast
        self.do_noise = do_noise
        self.do_blur = do_blur
        self.do_low_res = do_low_res
        
        # 增强参数
        self.rotation_angle_range = rotation_angle_range
        self.mirror_axes = mirror_axes
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_variance = noise_variance
        self.blur_sigma = blur_sigma
        self.low_res_scale = low_res_scale
        
        # 概率衰减参数
        self.initial_p = initial_p
        self.final_p = final_p
        self.max_epochs = max_epochs
        
        # 当前概率（初始化为initial_p）
        self.current_p = initial_p
    
    def compute_probability_disc(self, current_epoch):
        """
        计算当前epoch的增强概率（线性衰减）
        
        公式: p = initial_p + (final_p - initial_p) * (current_epoch / max_epochs)
        
        :param current_epoch: 当前epoch数
        :return: 当前增强概率
        """
        if current_epoch >= self.max_epochs:
            return self.final_p
        
        # 线性衰减
        progress = current_epoch / self.max_epochs
        p = self.initial_p + (self.final_p - self.initial_p) * progress
        
        return max(0.0, min(1.0, p))  # 确保在[0, 1]范围内
    
    def augment_rotation_disc(self, data, seg):
        """应用随机3D旋转，保持图像和标签的空间一致性"""
        if not self.do_rotation or np.random.rand() >= self.current_p:
            return data, seg
        
        angle = np.random.uniform(*self.rotation_angle_range)
        # 随机选择旋转平面
        axes_options = [(0, 1), (0, 2), (1, 2)]
        axes = axes_options[np.random.randint(0, 3)]
        
        augmented_data = np.zeros_like(data)
        for c in range(data.shape[0]):
            augmented_data[c] = rotate(data[c], angle, axes=axes, reshape=False, 
                                      order=3, mode='constant', cval=0)
        
        augmented_seg = np.zeros_like(seg)
        for c in range(seg.shape[0]):
            augmented_seg[c] = rotate(seg[c], angle, axes=axes, reshape=False, 
                                     order=0, mode='constant', cval=-1)
        
        return augmented_data, augmented_seg
    
    def augment_mirror_disc(self, data, seg):
        """应用随机镜像翻转"""
        if not self.do_mirror or np.random.rand() >= self.current_p:
            return data, seg
        
        # 添加形状检查和防御性编程
        if len(data.shape) < 4 or len(seg.shape) < 4:
            print(f"Warning: Unexpected shape in mirror augmentation. data: {data.shape}, seg: {seg.shape}")
            return data, seg
        
        # 随机选择翻转轴（+1是因为通道维度在前）
        axis_idx = np.random.randint(0, len(self.mirror_axes))
        axis = self.mirror_axes[axis_idx] + 1
        
        # 确保 axis 在有效范围内
        if axis >= len(data.shape) or axis >= len(seg.shape):
            print(f"Warning: axis {axis} out of range for shapes data: {data.shape}, seg: {seg.shape}")
            return data, seg
        
        # 确保该轴的维度大于0
        if data.shape[axis] == 0 or seg.shape[axis] == 0:
            print(f"Warning: Cannot flip along axis {axis}, dimension is 0")
            return data, seg
        
        try:
            data = np.flip(data, axis=axis).copy()
            seg = np.flip(seg, axis=axis).copy()
        except (ValueError, IndexError) as e:
            print(f"Warning: Mirror augmentation failed with error: {e}")
            print(f"  data.shape: {data.shape}, seg.shape: {seg.shape}, axis: {axis}")
            return data, seg
        
        return data, seg
    
    def augment_gamma_disc(self, data, seg):
        """应用gamma校正（仅对图像）"""
        if not self.do_gamma or np.random.rand() >= self.current_p:
            return data, seg
        
        gamma = np.random.uniform(*self.gamma_range)
        augmented_data = np.zeros_like(data)
        
        for c in range(data.shape[0]):
            channel_data = data[c]
            min_val = channel_data.min()
            max_val = channel_data.max()
            
            if max_val > min_val:
                # 归一化到[0, 1]
                normalized = (channel_data - min_val) / (max_val - min_val)
                # 应用gamma
                gamma_corrected = np.power(normalized, gamma)
                # 恢复到原始范围
                augmented_data[c] = gamma_corrected * (max_val - min_val) + min_val
            else:
                augmented_data[c] = channel_data
        
        return augmented_data, seg
    
    def augment_brightness_disc(self, data, seg):
        """调整亮度（添加常数）"""
        if not self.do_brightness or np.random.rand() >= self.current_p:
            return data, seg
        
        brightness_offset = np.random.uniform(*self.brightness_range)
        augmented_data = np.zeros_like(data)
        
        for c in range(data.shape[0]):
            channel_data = data[c]
            std_val = np.std(channel_data[channel_data > 0]) if np.any(channel_data > 0) else 1.0
            augmented_data[c] = channel_data + brightness_offset * std_val
        
        return augmented_data, seg
    
    def augment_contrast_disc(self, data, seg):
        """调整对比度（围绕均值相乘）"""
        if not self.do_contrast or np.random.rand() >= self.current_p:
            return data, seg
        
        contrast_factor = np.random.uniform(*self.contrast_range)
        augmented_data = np.zeros_like(data)
        
        for c in range(data.shape[0]):
            channel_data = data[c]
            non_zero_mask = channel_data > 0
            
            if np.any(non_zero_mask):
                mean_val = np.mean(channel_data[non_zero_mask])
                augmented_data[c] = (channel_data - mean_val) * contrast_factor + mean_val
            else:
                augmented_data[c] = channel_data
        
        return augmented_data, seg
    
    def augment_noise_disc(self, data, seg):
        """添加高斯噪声"""
        if not self.do_noise or np.random.rand() >= self.current_p:
            return data, seg
        
        variance = np.random.uniform(*self.noise_variance)
        augmented_data = np.zeros_like(data)
        
        for c in range(data.shape[0]):
            channel_data = data[c]
            std_val = np.std(channel_data[channel_data > 0]) if np.any(channel_data > 0) else 1.0
            noise = np.random.normal(0, variance * std_val, channel_data.shape)
            augmented_data[c] = channel_data + noise
        
        return augmented_data, seg
    
    def augment_blur_disc(self, data, seg):
        """应用高斯模糊"""
        if not self.do_blur or np.random.rand() >= self.current_p:
            return data, seg
        
        sigma = np.random.uniform(*self.blur_sigma)
        augmented_data = np.zeros_like(data)
        
        for c in range(data.shape[0]):
            augmented_data[c] = gaussian_filter(data[c], sigma=sigma)
        
        return augmented_data, seg
    
    def augment_low_resolution_disc(self, data, seg):
        """模拟低分辨率（下采样再上采样）"""
        if not self.do_low_res or np.random.rand() >= self.current_p:
            return data, seg
        
        scale = np.random.uniform(*self.low_res_scale)
        original_shape = data.shape[1:]  # (Z, X, Y)
        
        # 计算下采样尺寸
        down_shape = tuple(int(s * scale) for s in original_shape)
        
        # 使用PyTorch插值进行下采样和上采样
        augmented_data = torch_interpolate_3d_disc(data, down_shape, mode='trilinear', is_seg=False)
        augmented_data = torch_interpolate_3d_disc(augmented_data, original_shape, mode='trilinear', is_seg=False)
        
        augmented_seg = torch_interpolate_3d_disc(seg, down_shape, mode='nearest', is_seg=True)
        augmented_seg = torch_interpolate_3d_disc(augmented_seg, original_shape, mode='nearest', is_seg=True)
        
        return augmented_data, augmented_seg
    
    def __call__(self, data, seg, current_epoch=0):
        """
        应用数据增强流程
        
        :param data: 图像数据，shape (C, Z, X, Y)
        :param seg: 分割标签，shape (C, Z, X, Y)
        :param current_epoch: 当前训练epoch数，用于计算衰减概率
        :return: 增强后的数据和标签
        """
        # 更新当前概率
        self.current_p = self.compute_probability_disc(current_epoch)
        
        # 按顺序应用所有增强（每个增强独立判断是否执行）
        data, seg = self.augment_rotation_disc(data, seg)
        data, seg = self.augment_mirror_disc(data, seg)
        data, seg = self.augment_gamma_disc(data, seg)
        data, seg = self.augment_brightness_disc(data, seg)
        data, seg = self.augment_contrast_disc(data, seg)
        data, seg = self.augment_noise_disc(data, seg)
        data, seg = self.augment_blur_disc(data, seg)
        data, seg = self.augment_low_resolution_disc(data, seg)
        
        return data, seg
    
    def get_current_probability(self):
        """获取当前增强概率（用于日志记录）"""
        return self.current_p

