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

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnformer.utilities.nd_softmax import softmax_helper
from nnformer.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha: 类别权重，可以是标量、列表或张量。默认None表示所有类别权重相同
        gamma: 聚焦参数，默认2.0。gamma越大，对简单样本的抑制越强
               推荐范围: [0.5, 5.0], 常用值: 2.0
        reduction: 'mean', 'sum' or 'none'
        ignore_index: 忽略的标签索引
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # 处理 alpha 参数
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        
    def forward(self, input, target):
        """
        Args:
            input: (B, C, *) 网络输出 logits，未经过softmax
            target: (B, *) 或 (B, 1, *) 标签
        Returns:
            focal loss值
        """
        # 处理 target 维度，确保与 cross_entropy 兼容
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()
        
        # 获取输入的维度信息
        num_classes = input.shape[1]
        
        # 将 alpha 移到正确的设备
        if self.alpha is not None:
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            # 确保 alpha 的长度匹配类别数
            if len(self.alpha) == 1:
                alpha = self.alpha.expand(num_classes)
            else:
                alpha = self.alpha
        
        # 计算交叉熵损失（不进行reduction）
        ce_loss = F.cross_entropy(input, target, reduction='none', 
                                   ignore_index=self.ignore_index)
        
        # 计算 p_t：真实类别的预测概率
        p = F.softmax(input, dim=1)
        p_t = torch.exp(-ce_loss)  # 等价于 p[range(B), target]
        
        # 计算 focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用 alpha 权重
        if self.alpha is not None:
            # 为每个样本选择对应类别的 alpha
            # target 的形状可能是 (B, H, W, D) 等
            original_shape = target.shape
            target_flat = target.view(-1)
            
            # 获取每个位置对应的 alpha 值
            alpha_t = alpha[target_flat]
            alpha_t = alpha_t.view(original_shape)
            
            # 只对有效位置应用 alpha（排除 ignore_index）
            valid_mask = target != self.ignore_index
            focal_weight = torch.where(valid_mask, focal_weight * alpha_t, focal_weight)
        
        # 计算最终的 focal loss
        focal_loss = focal_weight * ce_loss
        
        # 应用 reduction
        if self.reduction == 'mean':
            valid_mask = target != self.ignore_index
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DC_and_Focal_loss(nn.Module):
    """
    Dice Loss + Focal Loss 组合损失函数
    适用于类别不平衡的医学图像分割任务
    
    Args:
        soft_dice_kwargs: SoftDiceLoss 的参数字典
        focal_kwargs: FocalLoss 的参数字典
        aggregate: 聚合方式，'sum' 表示两个损失相加
        square_dice: 是否使用平方 Dice Loss
        weight_focal: Focal Loss 的权重，默认1
        weight_dice: Dice Loss 的权重，默认1
        log_dice: 是否对 Dice Loss 取对数
    """
    def __init__(self, soft_dice_kwargs, focal_kwargs, aggregate="sum", 
                 square_dice=False, weight_focal=1, weight_dice=1, log_dice=False):
        """
        CAREFUL. Weights for Focal and Dice do not need to sum to one. 
        You can set whatever you want.
        """
        super(DC_and_Focal_loss, self).__init__()
        
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.aggregate = aggregate
        
        # 初始化 Focal Loss
        self.focal = FocalLoss(**focal_kwargs)
        
        # 初始化 Dice Loss
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output: 网络输出，形状 (B, num_classes, *)
        :param target: 目标标签，形状 (B, 1, *)
        :return: 组合损失值
        """
        # 计算 Dice Loss
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)
        
        # 计算 Focal Loss
        # Focal Loss 需要 (B, C, *) 的输入和 (B, *) 或 (B, 1, *) 的 target
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 else 0
        
        # 聚合两个损失
        if self.aggregate == "sum":
            result = self.weight_focal * focal_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("Only 'sum' aggregate is implemented")
        
        return result


class DC_and_Focal_loss_with_ignore(nn.Module):
    """
    带 ignore_label 支持的 Dice + Focal Loss
    用于处理需要忽略某些标签的情况
    """
    def __init__(self, soft_dice_kwargs, focal_kwargs, aggregate="sum", 
                 square_dice=False, weight_focal=1, weight_dice=1, 
                 log_dice=False, ignore_label=None):
        super(DC_and_Focal_loss_with_ignore, self).__init__()
        
        if ignore_label is not None:
            assert not square_dice, 'ignore_label with square_dice not implemented'
            # 确保 focal loss 知道要忽略的标签
            focal_kwargs['ignore_index'] = ignore_label
        
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.aggregate = aggregate
        self.ignore_label = ignore_label
        
        # 初始化 Focal Loss
        self.focal = FocalLoss(**focal_kwargs)
        
        # 初始化 Dice Loss
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        # 处理 ignore_label
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target_masked = target.clone()
            target_masked[~mask] = 0
            mask = mask.float()
        else:
            mask = None
            target_masked = target
        
        # 计算 Dice Loss
        dc_loss = self.dc(net_output, target_masked, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)
        
        # 计算 Focal Loss（已经在 FocalLoss 中处理 ignore_index）
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 else 0
        
        # 聚合
        if self.aggregate == "sum":
            result = self.weight_focal * focal_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("Only 'sum' aggregate is implemented")
        
        return result

