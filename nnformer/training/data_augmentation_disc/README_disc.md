# 腰椎间盘退变MRI数据增强模块

## 📋 概述

本模块为腰椎间盘退变MRI图像提供专用的3D数据增强功能，具有以下特点：

- ✅ **8种增强技术**：旋转、镜像、Gamma、亮度、对比度、噪声、模糊、低分辨率
- ✅ **保守参数设计**：保持脊柱解剖结构的有效性
- ✅ **空间一致性**：图像和标签同步增强
- ✅ **概率线性衰减**：增强概率随epoch自动递减
- ✅ **训练时动态应用**：每个epoch看到不同的增强版本

## 🎯 设计理念

### 为什么在训练时增强？

1. **最大化数据多样性**：每个epoch都是不同的增强版本
2. **节省存储空间**：不需要保存多个增强副本
3. **灵活调整策略**：可随时修改增强参数
4. **符合最佳实践**：医学图像分析的标准流程

### 概率衰减策略

```
概率 = initial_p + (final_p - initial_p) * (current_epoch / max_epochs)

默认配置：
- initial_p = 0.2  (20% 在epoch 0)
- final_p = 0.05   (5% 在epoch 1000)
- 线性递减
```

**原理**：
- 训练初期：强增强帮助模型快速学习多样性
- 训练后期：弱增强让模型精细调优

## 🚀 使用方法

### 1. 预处理阶段（cropping.py）

预处理只做基础操作，**不做增强**：
```python
# 在 ImageCropper 中
1. 读取NIfTI文件
2. PyTorch三维插值重采样 → (512, 512, 85)
3. 中心裁剪 → (85, 256, 216)
4. 强度归一化
5. 保存为.npz文件
```

### 2. 训练阶段（nnFormerTrainerV2_nnformer_disc.py）

训练时动态应用增强：

```python
# 在 run_iteration 方法中
1. 加载.npz数据
2. 对每个batch样本应用增强（自动根据epoch调整概率）
3. 送入网络训练
```

### 3. 验证/测试阶段

**自动禁用增强**：
- `run_iteration` 的 `do_backprop=False` 时不增强
- 确保验证和测试使用原始数据

## 📊 增强技术详解

### 1. 随机旋转 (Random 3D Rotation)
```python
rotation_angle_range=(-15, 15)  # ±15度
```
- 随机选择3D旋转平面：(Z,X), (Z,Y), (X,Y)
- 保守角度范围，保护脊柱结构
- 图像：三次样条插值 (order=3)
- 标签：最近邻插值 (order=0)

### 2. 镜像翻转 (Mirror Flipping)
```python
mirror_axes=(0, 1, 2)  # Z, X, Y轴
```
- 随机选择一个轴进行翻转
- 适用于脊柱的左右对称性
- 注意：Z轴翻转会改变上下方向

### 3. Gamma校正 (Gamma Correction)
```python
gamma_range=(0.7, 1.5)
```
- 非线性对比度调整
- 模拟不同的MRI采集参数
- 仅作用于图像，不影响标签

### 4. 亮度调整 (Brightness Adjustment)
```python
brightness_range=(-0.2, 0.2)  # 相对于标准差
```
- 添加/减去常数（按图像标准差缩放）
- 模拟扫描仪校准差异

### 5. 对比度调整 (Contrast Adjustment)
```python
contrast_range=(0.75, 1.25)  # 乘数因子
```
- 围绕均值相乘
- 增强/减弱组织对比度

### 6. 高斯噪声 (Gaussian Noise)
```python
noise_variance=(0, 0.05)  # 相对于标准差
```
- 添加真实的扫描仪噪声
- 提高模型鲁棒性

### 7. 高斯模糊 (Gaussian Blur)
```python
blur_sigma=(0.5, 1.5)
```
- 模拟轻微的运动模糊
- 减少对高频细节的过拟合

### 8. 低分辨率模拟 (Low Resolution Simulation)
```python
low_res_scale=(0.5, 1.0)
```
- 下采样再上采样
- 模拟不同分辨率的扫描
- 使用PyTorch插值（GPU加速）

## ⚙️ 配置参数

### 当前默认配置（nnFormerTrainerV2_nnformer_disc.py）

```python
self.disc_augmentor = DataAugmentation3D_disc(
    # 增强开关（全部启用）
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
    initial_p=0.2,           # 20%
    final_p=0.05,            # 5%
    max_epochs=1000
)
```

### 修改配置

如需调整，修改 `nnFormerTrainerV2_nnformer_disc.py` 的 `__init__` 方法：

```python
# 例如：更保守的增强
self.disc_augmentor = DataAugmentation3D_disc(
    rotation_angle_range=(-10, 10),    # 更小角度
    mirror_axes=(1, 2),                 # 不翻转Z轴
    initial_p=0.15,                     # 更低初始概率
    final_p=0.02,                       # 更低最终概率
    max_epochs=self.max_num_epochs
)

# 或者：禁用某些增强
self.disc_augmentor = DataAugmentation3D_disc(
    do_rotation=True,
    do_mirror=True,
    do_gamma=False,        # 禁用gamma
    do_brightness=False,   # 禁用亮度
    do_contrast=True,
    do_noise=True,
    do_blur=False,         # 禁用模糊
    do_low_res=False,      # 禁用低分辨率
    ...
)
```

### 完全禁用增强

在 `nnFormerTrainerV2_nnformer_disc.py` 的 `__init__` 中：
```python
self.use_disc_augmentation = False  # 设为False
```

## 📈 训练监控

### 日志输出

训练时会自动记录当前的增强概率：
```
Epoch 0:   Disc augmentation probability: 0.2000
Epoch 100: Disc augmentation probability: 0.1700
Epoch 500: Disc augmentation probability: 0.1250
Epoch 999: Disc augmentation probability: 0.0500
```

### 可视化概率衰减

```python
import matplotlib.pyplot as plt
import numpy as np

# 模拟1000个epoch的概率变化
epochs = np.arange(1000)
initial_p = 0.2
final_p = 0.05
prob = initial_p + (final_p - initial_p) * (epochs / 1000)

plt.plot(epochs, prob)
plt.xlabel('Epoch')
plt.ylabel('Augmentation Probability')
plt.title('Augmentation Probability Decay')
plt.grid(True)
plt.show()
```

## 🔧 技术细节

### 数据流程

```
训练数据流:
┌──────────────┐
│  .npz文件    │ (已归一化)
│ (85,256,216) │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ DataLoader   │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────┐
│ run_iteration()              │
│ for each sample in batch:   │
│   if training:               │
│     应用增强(根据epoch调整p) │
└──────┬───────────────────────┘
       │
       ↓
┌──────────────┐
│  to_torch()  │
│  to_cuda()   │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│   Network    │
└──────────────┘
```

### 性能考虑

1. **CPU vs GPU**：增强在CPU上执行（数据加载阶段）
2. **速度**：旋转最慢，其他增强很快
3. **内存**：按样本处理，不会显著增加内存

### 概率计算

每个增强技术独立判断是否应用：
```python
if np.random.rand() < current_p:
    # 应用该增强
```

这意味着一个样本可能同时应用多个增强，也可能一个都不应用。

## ✅ 验证检查

### 确认增强生效

1. 查看训练日志，确认有概率输出
2. 第一个epoch应该显示 `0.2000`
3. 随epoch增加，概率应该线性下降

### 确认空间一致性

增强器自动保证：
- 图像和标签同步变换
- 标签值保持不变（只有空间位置变化）
- 背景区域正确处理

## 🐛 故障排除

### 问题1：增强未生效
**检查**：
```python
self.use_disc_augmentation = True  # 确认为True
```

### 问题2：训练过慢
**解决**：
```python
# 禁用旋转和低分辨率（最慢的两个）
do_rotation=False,
do_low_res=False,
```

### 问题3：模型不收敛
**可能原因**：增强太强
**解决**：
```python
# 降低初始概率
initial_p=0.1,  # 从0.2降到0.1
```

### 问题4：验证集也被增强了
**检查**：确认 `run_iteration` 中的条件判断
```python
if do_backprop and self.use_disc_augmentation:
```
`do_backprop=False` 时不应增强。

## 📚 文件结构

```
nnformer/
├── preprocessing/
│   └── cropping.py              # 预处理（重采样+裁剪+归一化）
│
└── training/
    ├── data_augmentation_disc/
    │   ├── __init__.py
    │   ├── augmentation_disc.py # 增强实现
    │   └── README_disc.md       # 本文档
    │
    └── network_training/
        └── nnFormerTrainerV2_nnformer_disc.py  # 训练器（集成增强）
```

## 📝 引用

如果使用本增强模块，请引用：
- 原始nnFormer论文
- PyTorch和相关库

## 📞 支持

如有问题或建议，请：
1. 查看本README
2. 检查训练日志
3. 调整增强参数

---

**版本**: 1.0  
**更新**: 2025年10月  
**适用**: 腰椎间盘退变MRI分析

