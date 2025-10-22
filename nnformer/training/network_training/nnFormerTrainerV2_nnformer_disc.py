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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture.nnFormer_disc import nnFormer
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer import nnFormerTrainer
from nnformer.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, default_3D_augmentation_params, get_patch_size


class DiscSegmentationLoss(nn.Module):
    """
    专门针对椎间盘分割优化的损失函数
    - 结合Dice Loss和Focal Loss
    - 处理类别不平衡问题
    - 增强对边界细节的关注
    """
    def __init__(self, num_classes=3, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(DiscSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # 椎间盘分割的类别权重：背景、正常、退变
        self.class_weights = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float32)  # 退变椎间盘权重更高
        
        # Dice Loss
        self.dice_loss = SoftDiceLoss(apply_nonlin=softmax_helper, 
                                     batch_dice=True, 
                                     smooth=1e-5, 
                                     do_bg=False)
        
    def focal_loss(self, inputs, targets):
        """Focal Loss for addressing class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights.to(inputs.device))
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, inputs, targets):
        # 确保targets是long类型
        if targets.dtype != torch.long:
            targets = targets.long()
        
        # 如果targets是one-hot，转换为类别索引
        if len(targets.shape) == 5:  # (B, C, D, H, W)
            targets = torch.argmax(targets, dim=1)
        
        # Focal Loss
        focal = self.focal_loss(inputs, targets)
        
        # Dice Loss
        dice = self.dice_loss(inputs, targets)
        
        # 组合损失
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss


class nnFormerTrainerV2_nnformer_disc(nnFormerTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        # 【修改】：降低初始学习率从 5e-3 到 2e-3，更稳定的训练
        # 椎间盘退变分割是细粒度任务，需要更谨慎的学习率
        self.initial_lr = 2e-3
        self.warmup_epochs = 50  # 学习率warmup轮数
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight=False
        
        self.load_plans_file()    
        
        if len(self.plans['plans_per_stage'])==2:
            Stage=1
        else:
            Stage=0
        
        self.crop_size=self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels=self.plans['num_modalities']
        self.num_classes=self.plans['num_classes'] + 1
        self.conv_op=nn.Conv3d
        
        self.embedding_dim=96
        self.depths=[2, 2, 2, 2]
        self.num_heads=[3, 6, 12, 24]
        self.embedding_patch_size=[1,4,4]  # 保持原始patch size
        self.window_size=[[3,5,5],[3,5,5],[7,10,10],[3,5,5]]
        # 【修正】：使用标准的下采样策略，确保与 nnFormer 兼容
        # 基于 crop_size=[85, 216, 256] 和 embedding_patch_size=[1,4,4]
        # 使用标准的 2x2x2 下采样，这是 nnFormer 的默认策略
        self.down_stride=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        self.deep_supervision=False  # 【关键优化】：启用深度监督提升训练效果
        
        # 【新增】：使用标准的 pool_op_kernel_sizes
        # 这是 nnFormer 的标准配置，确保完全兼容
        self.net_num_pool_op_kernel_sizes = [
            [2, 2, 2],  # Stage 1: 标准 2x2x2 下采样
            [2, 2, 2],  # Stage 2: 标准 2x2x2 下采样
            [2, 2, 2],  # Stage 3: 标准 2x2x2 下采样
            [2, 2, 2]   # Stage 4: 标准 2x2x2 下采样
        ]
        
        # 对应的卷积核尺寸
        self.net_conv_kernel_sizes = [[3, 3, 3]] * (len(self.net_num_pool_op_kernel_sizes) + 1)
        
        # 【关键优化】：使用专门针对椎间盘分割优化的损失函数
        self.loss = DiscSegmentationLoss(num_classes=3, alpha=0.25, gamma=2.0, 
                                        dice_weight=1.0, focal_weight=0.5)
        
    def initialize(self, training=True, force_load_plans=False):
        """
        【简化版本】：移除所有数据增强相关代码，专注核心训练逻辑
        - 数据增强已在预处理阶段完成
        - 直接使用基础数据加载器，无动态增强

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.process_plans(self.plans)
            
            # 设置数据增强参数
            self.setup_DA_params()

            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                #mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                #weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################
            
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +"_stage%d" % self.stage)
                         
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # 【简化】：直接使用基础数据加载器，无任何增强
                self.tr_gen = self.dl_tr
                self.val_gen = self.dl_val
                
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("数据增强已在预处理阶段完成，训练时直接使用原始数据")
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """ 
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
  
      
        
        self.network=nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                down_stride=self.down_stride,
                                deep_supervision=self.deep_supervision)
        # if self.load_pretrain_weight:
        #     checkpoint = torch.load("/home/xychen/jsguo/weight/tumor_pretrain.model", map_location='cpu') # acdc and tumor use the same pretrain weight
        #     ck={}
            
        #     for i in self.network.state_dict():
        #         if i in checkpoint:
        #             print(i)
        #             ck.update({i:checkpoint[i]})
        #         else:
        #             ck.update({i:self.network.state_dict()[i]})
        #     self.network.load_state_dict(ck)
        #     print('I am using the pre_train weight!!')
        
     
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True,
                                                         use_tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        【优化】：添加测试时增强（TTA）提升推理效果
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        
        if use_tta:
            # 测试时增强：多次预测并平均
            predictions = []
            
            # 原始预测
            pred_orig = super().predict_preprocessed_data_return_seg_and_softmax(
                data, do_mirroring=False, mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window, step_size=step_size, 
                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                mixed_precision=mixed_precision)
            predictions.append(pred_orig[1])  # softmax
            
            # 翻转增强
            if do_mirroring:
                for axis in [0, 1, 2]:  # 沿每个轴翻转
                    data_flipped = np.flip(data, axis=axis)
                    pred_flipped = super().predict_preprocessed_data_return_seg_and_softmax(
                        data_flipped, do_mirroring=False, mirror_axes=mirror_axes,
                        use_sliding_window=use_sliding_window, step_size=step_size,
                        use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                        pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=False,
                        mixed_precision=mixed_precision)
                    # 翻转回来
                    pred_flipped_back = np.flip(pred_flipped[1], axis=axis)
                    predictions.append(pred_flipped_back)
            
            # 平均所有预测
            avg_softmax = np.mean(predictions, axis=0)
            seg = np.argmax(avg_softmax, axis=0).astype(np.uint8)
            ret = (seg, avg_softmax)
        else:
            # 标准预测
            ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                           do_mirroring=do_mirroring,
                                                                           mirror_axes=mirror_axes,
                                                                           use_sliding_window=use_sliding_window,
                                                                           step_size=step_size, use_gaussian=use_gaussian,
                                                                           pad_border_mode=pad_border_mode,
                                                                           pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                           verbose=verbose,
                                                                           mixed_precision=mixed_precision)
        
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        # 处理数据字典中的键名差异：DataLoader3D返回'seg'，增强器返回'target'
        target = data_dict.get('target', data_dict.get('seg'))

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            splits[self.fold]['train'] = np.array([
                '100niquanjin20130610_interp', '101shengxiaolong20130504_interp', '102gaopeijun20130220_interp',
                '102gaopeijun20150309_interp', '103wanghuiming20140502_interp', '103wanghuiming20150112_interp',
                '105wangqunfeng20180104_interp', '106shenpeifeng20140604_interp', '106shenpeifeng20150201_interp',
                '107shenjinfang20140623_interp', '108zhufalin20150417_interp', '109feidonglin20130611_interp',
                '109feidonglin20160515_interp', '111shenyongfeng20120922_interp', '112wuchengying20171018_interp',
                '113yangerbao20120213_interp', '114zhanghua20120309_interp', '114zhanghua20130330_interp',
                '115huangjinyun20180320_interp', '116wangyongming20120921_interp', '117yelei20160324_interp',
                '118liujialiang20120522_interp', '118liujialiang20121226_interp', '119lixiaozhen20120716_interp',
                '11qianjianliang20171030_interp', '11qianjianliang20180202_interp', '121zhangfugen20120830_interp',
                '121zhangfugen20140416_interp', '122yangyijun20120924_interp', '122yangyijun20130130_interp',
                '123zhanghongjuan20150916_interp', '124panyong20121109_interp', '124panyong20181005_interp',
                '125sunxiaoming20121204_interp', '125sunxiaoming20150429_interp', '126zhuhaigen20120922_interp',
                '126zhuhaigen20130104_interp', '127zhangyuhua20130116_interp', '127zhangyuhua20211009_interp',
                '128sunxueying20130324_interp', '128sunxueying20240201_interp', '129zhouyi20131225_interp',
                '12yudunyao20110124_interp', '12yudunyao20140809_interp', '132liangxiaohu20140416_interp',
                '132liangxiaohu20140819_interp', '133guyongqing20140420_interp', '134zhouchunmei20140430_interp',
                '138dingying20140824_interp', '138dingying20150903_interp', '139wanglong20110504_interp',
                '139wanglong20141009_interp', '13zhoujie20160712_interp', '140luojuying20141104_interp',
                '140luojuying20190104_interp', '141wangna20140716_interp', '141wangna20150109_interp',
                '142haozhonglu20141204_interp', '142haozhonglu20150107_interp', '143gaoweimin20150108_interp',
                '144tangxiaomei20150325_interp', '145maobowen20151013_interp', '146lishengjun20150823_interp',
                '146lishengjun20150903_interp', '147yangjianqiang20150919_interp', '148lutao20151004_interp',
                '148lutao20201206_interp', '149zhoudingyong20151231_interp', '149zhoudingyong20160406_interp',
                '14liujinmin20110618_interp', '14liujinmin20120911_interp', '150tanghuinan20151219_interp',
                '150tanghuinan20160205_interp', '151liping20160114_interp', '151liping20170423_interp',
                '153xujun20160804_interp', '154luxueliang20160314_interp', '154luxueliang20160526_interp',
                '155chenjinquan20160328_interp', '155chenjinquan20160504_interp', '156zhouyong20120723_interp',
                '156zhouyong20120813_interp', '157wangyan20160407_interp', '157wangyan20160530_interp',
                '158yaoyong20120505_interp', '159wangzhenyong20180831_interp', '15chaixiaodong20150622_interp',
                '15chaixiaodong20160418_interp', '160shichengli20160724_interp', '160shichengli20170302_interp',
                '161chenjiabin20180613_interp', '162gechangmei20160927_interp', '162gechangmei20170326_interp',
                '163xuwang20161030_interp', '163xuwang20240406_interp', '164luoguangyao20170109_interp',
                '164luoguangyao20170219_interp', '165xiewen20170117_interp', '166yangtingting20170120_interp',
                '167yuanyuechun20170213_interp', '167yuanyuechun20170531_interp', '168fangming20170215_interp',
                '169luyongqing20180622_interp', '169luyongqing20200826_interp', '16mayanjun20130510_interp',
                '16mayanjun20180116_interp', '170shenguizhen20190203_interp', '171guxingming20170401_interp',
                '172liujieping20170416_interp', '173huangyuhua20170528_interp', '173huangyuhua20170705_interp',
                '174shijianming20170610_interp', '174shijianming20170718_interp', '175wulanfang20160812_interp',
                '176liuxuelong20170710_interp', '178wangzhiming20180609_interp', '179sunshiliang20170727_interp',
                '17wangmingzhi20150504_interp', '17wangmingzhi20151222_interp', '180tangfuzhen20170802_interp',
                '180tangfuzhen20230516_interp', '181chenyong20170811_interp', '183chenqiubo20161116_interp',
                '183chenqiubo20180310_interp', '184xueting20171020_interp', '185huangliya20171029_interp',
                '185huangliya20180206_interp', '186wuxiaodong20171107_interp', '187yaozhichun20171108_interp',
                '187yaozhichun20180524_interp', '188zhanghong20171108_interp', '188zhanghong20180101_interp',
                '189xulinbo20171009_interp', '189xulinbo20171115_interp', '18liujiaran20151102_interp',
                '18liujiaran20160629_interp', '190tangxueyun20171129_interp', '191luguoming20180814_interp',
                '192jingrongjuan20171231_interp', '192jingrongjuan20220705_interp', '193fanzhenhua20200102_interp',
                '194fangzhenmei20231202_interp', '195lumeizhen20180414_interp', '195lumeizhen20230629_interp',
                '196xixingyan20180415_interp', '196xixingyan20180521_interp', '197zhangliying20161228_interp',
                '198chenkai20160731_interp', '198chenkai20190831_interp', '199wenruiping20180625_interp',
                '199wenruiping20200928_interp', '19wuminhong20161228_interp', '1lujianhua20130316_interp',
                '1lujianhua20130724_interp', '200wangcunmao20180630_interp', '201zhugang20180419001201_interp',
                '201zhugang20180704000606_interp', '202zhouzhaozhen20180528000941_interp',
                '202zhouzhaozhen20190903000059_interp', '203yangweifang20170411000560_interp',
                '203yangweifang20180724001249_interp', '204yuanfenge20180813000874_interp',
                '205lulongmei20161012000407_interp', '205lulongmei20180906000816_interp',
                '206sunlijuan20180917001241_interp', '207libianli20180922000587_interp',
                '208zhangxiaofang20181009000555_interp', '209mayujuan20160805000686_interp',
                '209mayujuan20181110000801_interp', '20duxiaohu20170105_interp', '20duxiaohu20171002_interp',
                '212zhusannan20140320000680_interp', '213zhangqingzhen20181214001009_interp',
                '214dingjiawang20180923000173_interp', '215xuhaiyan20190127000543_interp',
                '217lixiaoxing20190822001363_interp', '218kongjiahao20170221000055_interp',
                '218kongjiahao20221123000688_interp', '21shaoyue20181225_interp', '220tangyubao20220208000538_interp',
                '221songwei20170906000579_interp', '221songwei20171228000693_interp', '224yanjiajing20180711000462_interp',
                '227wangxingang20180122000213_interp', '228shifue20180502000401_interp', '228shifue20190612000527_interp',
                '22shenzhaohong20170424_interp', '22shenzhaohong20181212_interp', '230lishuanghua20180421000450_interp',
                '231jingqingxu20190711001364_interp', '231jingqingxu20191105000429_interp', '232wangxiuying20181027000730_interp',
                '232wangxiuying20210210000374_interp', '233wangjian20190408000806_interp', '235qizonglan20180404000617_interp',
                '236yaochunjiang20160824000748_interp', '236yaochunjiang20231118000955_interp', '239wangjianzhen20170614000691_interp',
                '239wangjianzhen20170906000789_interp', '23zhangjunhua20181114_interp', '23zhangjunhua20190612_interp',
                '240chengliang20180828000778_interp', '240chengliang20190520000658_interp', '241zhoulifang20191104000501_interp',
                '243wuwenjie20190716000383_interp', '243wuwenjie20191022001192_interp', '244shenyinyun20190514000572_interp',
                '244shenyinyun20190816000206_interp', '245jinminxia20190920001238_interp', '245jinminxia20220624000898_interp',
                '247sundani20190211001167_interp', '248jinjunyu20191009000152_interp', '249shenjianxin20191010001435_interp',
                '249shenjianxin20200725000588_interp', '24chentaiwen20120222_interp', '250lizhiqin20180514000529_interp',
                '250lizhiqin20221031000960_interp', '251chenjianjun20191101000641_interp', '252zhouchengcheng20200413000214_interp',
                '254zhaoqingtao20191125001091_interp', '255kexilong20190416000284_interp', '255kexilong20191126000301_interp',
                '256shenmeifang20190417000811_interp', '256shenmeifang20210409000542_interp', '258sunwencui20191213000961_interp',
                '258sunwencui20231205000997_interp', '259feiweiqiang20180409000752_interp', '25juyuxiang20130918_interp',
                '25juyuxiang20240127_interp', '260zhangsheng20180520000336_interp', '260zhangsheng20181114000791_interp',
                '261wuqinhua20190221001263_interp', '262xiaoqing20181217001162_interp', '262xiaoqing20190528001246_interp',
                '263suichunhui20190806000741_interp', '264zhudongxue20181115000497_interp', '264zhudongxue20210420001118_interp',
                '265yangyouchun20181005000403_interp', '265yangyouchun20190107000905_interp', '266mouwenliang20190214000415_interp',
                '266mouwenliang20190606001232_interp', '267zhangjianfeng20181210000760_interp', '268wangshili20191019000117_interp',
                '269lijianzhong20170126000196_interp', '26jinhao20141118_interp', '26jinhao20150506_interp', '270sunliyang20160502000516_interp',
                '271menggang20190616000428_interp', '272liuwentian20170104000641_interp', '272liuwentian20170426001159_interp',
                '273yangyongqing20171229000326_interp', '274dailiangyu20190324000659_interp', '274dailiangyu20220313000333_interp',
                '276yekaiyuan20180711001178_interp', '277zhouyichen20181227000793_interp', '278guruiying20180907000749_interp',
                '279luoguoping20181024000959_interp', '27chenjiawei20141115_interp', '27chenjiawei20150109_interp',
                '280zhenglijian20190115000643_interp', '281yuqinyun20190211001265_interp', '282sunxueliang20190322000955_interp',
                '284minsai20190926000135_interp', '285banzheng20190929000159_interp', '286xiyungen20191031000615_interp',
                '287wuyaqing20191105000764_interp', '289xiarong20200721000348_interp', '28luxueyun20140925_interp',
                '290hushengshu20200114000669_interp', '290hushengshu20200828000475_interp', '291yeliping20240707000337_interp',
                '293zhangyongliang20170425001215_interp', '294zhouzhendong20150305000616_interp', '294zhouzhendong20240708001688_interp',
                '296tangxiuxiang20170707000456_interp', '29caoqiang20150606_interp', '29caoqiang20180504_interp', '2yexin20170313_interp',
                '2yexin20170605_interp', '301zhangjianxun20180212_interp', '301zhangjianxun20180619_interp', '302zhoujinbao20180817_interp',
                '303xugenmei20170319_interp', '303xugenmei20180913_interp', '304liuhongyan20180724_interp', '305zhanglixin20181030_interp',
                '305zhanglixin20190307_interp', '306wangyinhai20181115_interp', '307chenfangzhen20181128_interp', '308xuyi20210315_interp',
                '308xuyi20240727_interp', '30chaizhendong20150106_interp', '30chaizhendong20170807_interp', '310mengchenghua20111129_interp',
                '311wangjianping20130411_interp', '312tumin20120228_interp', '313liuyong20150314_interp', '314liqing20110901_interp',
                '317majianfen20121229_interp', '317majianfen20130525_interp', '318jinhuijuan20150427000841_interp', '319zhanglu20150731_interp',
                '31yanyuqin20110221_interp', '31yanyuqin20120919_interp', '320zhoukaifeng20120226_interp', '321zhangjinrong20110916_interp',
                '322tangxinggen20110922_interp', '323xielijuan20120715_interp', '324linwenfeng20120725_interp', '325wangxu20120727_interp',
                '326wangxiaoxin20140904_interp', '327zhouhongshan20130529_interp', '328haolianxi20130807_interp', '329xuwenbing20131012_interp',
                '32zhudazhou20140107_interp', '330zhoujuan20131210_interp', '331yuanxinmin20131127_interp', '332shenjianqiang20131202_interp',
                '333huyongping20140112_interp', '334hefan20140107_interp', '335youfengying20140216_interp', '336tangwei20140310_interp',
                '337lizhenfa20140410_interp', '338zhaoxia20140604_interp', '339moqiulin20140804_interp', '33zhaoxiumei20120322_interp',
                '340xuailan20141028_interp', '341chenjianchun20141116_interp', '342yangguanjie20150507_interp', '343zhangwenhua20150506_interp',
                '344wuqing20150509_interp', '346wuzhifang20150523_interp', '347lichaoliang20150720_interp', '348wudarong20150730_interp',
                '349zhangshufang20150930_interp', '350gaoqing20151031_interp', '351lihebao20151023_interp', '354liyang20210521_interp',
                '356chenzengguang20230220_interp', '357zhangfenglei20190124_interp', '359chencheng20211013_interp', '35qujiqing20170901_interp',
                '361chenhaoshen20211109_interp', '362jiangwei20230811_interp', '362jiangwei20231106_interp', '363caizhaoxia20230818_interp',
                '363caizhaoxia20230913_interp', '364wujialin1次20190829_interp', '365liuzijin20230830_interp', '366lizhihua20230912_interp',
                '367zhangyuewu20240225_interp', '368daiyangyang20200406_interp', '36ruijinjin20180111_interp', '370maguosheng20240308_interp',
                '371zuoguanghua20240314_interp', '372wangqing20240804_interp', '373zhangdi20230920_interp', '374chenchen20240423_interp',
                '374chenchen20240730_interp', '375caofanyuan20240415_interp', '375caofanyuan20240702_interp', '376xujing20241116_interp',
                '377weiluchun20221111_interp', '377weiluchun20240614_interp', '378wujingxiao20181212_interp', '37shenyang20171120_interp',
                '380xuyunan20240719_interp', '381zhangjianguo20240815_interp', '382taoguoliang20240708_interp', '383qiuli20200910_interp',
                '383qiuli20240829_interp', '384hongqinbao20241023_interp', '38chenyongkang20200516_interp', '392malingyuan20210613_interp',
                '39liqiu20181001_interp', '39liqiu20210402_interp', '3zhuxueliang20161202_interp', '3zhuxueliang20170215_interp',
                '40yinxingya20181107_interp', '40yinxingya20191028_interp', '41yuanxiu20150824_interp', '41yuanxiu20150827_interp',
                '42liuqian20120718_interp', '43xiping20120221_interp', '43xiping20120919_interp', '44niyongguang20120112_interp',
                '44niyongguang20140722_interp', '45qianyufang20140522_interp', '45qianyufang20140720_interp', '46hejian20130107_interp',
                '46hejian20130525_interp', '47feiyongheng20110216_interp', '47feiyongheng20111219_interp', '49xuzhengde20130630_interp',
                '49xuzhengde20130909_interp', '4yangxiaofeng20150107_interp', '4yangxiaofeng20191012_interp', '50wufang20110114_interp'
            ])
            splits[self.fold]['val'] = np.array([
               '50wufang20170525_interp', '51liuyi20130701_interp', '51liuyi20131007_interp', '52yuanzhengang20120409_interp',
                '52yuanzhengang20120523_interp', '53niqian20131102_interp', '53niqian20191206_interp', '54chenchunhui20130908_interp',
                '55zhulan20120210_interp', '55zhulan20160624_interp', '56zhangshenghui20150507_interp', '57zhangsonglin20140710_interp',
                '57zhangsonglin20140825_interp', '58zhengxiaoqing20140628_interp', '59jiangfengjin20221102_interp',
                '5luominglan20150819_interp', '5luominglan20151114_interp', '60zhanghong20120807_interp', '60zhanghong20151211_interp',
                '61luxiujin20170701_interp', '61luxiujin20190311_interp', '62huanglingfang20110829_interp', '63maoyiping20110615_interp',
                '63maoyiping20150317_interp', '64zhangxiaohong20110701_interp', '64zhangxiaohong20111103_interp', '65guoruizhi20110926_interp',
                '65guoruizhi20120604_interp', '66xuqianhua20120119_interp', '66xuqianhua20150302_interp', '67liyuezhen20120201_interp',
                '67liyuezhen20121128_interp', '68fengxian20120316_interp', '68fengxian20120818_interp', '69yuliqing20110902_interp',
                '6geyijiang20170607_interp', '6geyijiang20170909_interp', '70sunxiaokuan20110708_interp', '70sunxiaokuan20170326_interp',
                '71zhouxiaojin20120318_interp', '72zhengjiajun20120530_interp', '72zhengjiajun20121019_interp', '73tangwenjuan20120804_interp',
                '73tangwenjuan20121207_interp', '74wuyifan20121004_interp', '75huangfuming20120823_interp', '76wujian20130416_interp',
                '76wujian20130723_interp', '77zhangwenmei20130523_interp', '77zhangwenmei20130902_interp', '78wukang20130601_interp',
                '78wukang20140301_interp', '79lichaoyi20140119_interp', '7huangzhijun20170110_interp', '7huangzhijun20170415_interp',
                '80mahongliang20140113_interp', '81chenqian20130804_interp', '81chenqian20131117_interp', '82shenshuigen20131212_interp',
                '84sunwei20140519_interp', '85shenqichun20140723_interp', '86xuailong20140802_interp', '86xuailong20150224_interp',
                '87zhangyan20140829_interp', '87zhangyan20141030_interp', '88chenzhi20140609_interp', '89caozhengkang20151015_interp',
                '8puchunquan20161101_interp', '90xiwenqing20140117_interp', '90xiwenqing20151121_interp', '92wanggonglin20150614_interp',
                '93zhangcaizhu20120906_interp', '93zhangcaizhu20121029_interp', '94lifengmei20120807_interp', '95zhangfan20120616_interp',
                '95zhangfan20160415_interp', '97zhoulei20120115_interp', '97zhoulei20130519_interp', '98hewei20131109_interp',
                '98hewei20140317_interp', '99gujinfeng20131228_interp', '9zhengkefu20150523_interp', '9zhengkefu20160706_interp'
            ])
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]


    def maybe_update_lr(self, epoch=None):
        """
        【优化】：实现学习率warmup和更精细的调度策略
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
            
        # Warmup阶段：线性增长到初始学习率
        if ep <= self.warmup_epochs:
            lr = self.initial_lr * (ep / self.warmup_epochs)
        else:
            # Warmup后使用poly学习率衰减
            lr = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            
        self.optimizer.param_groups[0]['lr'] = lr
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret

    def setup_DA_params(self):
        """
        设置数据增强参数，包括basic_generator_patch_size
        """
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2
