# NNFormer：通过3D变压器进行体积医学图像分割

在2022/02/11，我们重建了NNFormer的代码以匹配最新报告的性能[draft](https://arxiv.org/abs/2109.03201)。新代码产生的结果更稳定，因此更容易复制！

在2022/02/26，我们在文件nnformer/run_training.py中添加种子，并set torch.backends.cudnn.benchmark和torch.backends.cudnn.sabled nabled nabled nabled sualbleds以提高效率。

---
## 安装
#### 1。系统要求
我们在运行Ubuntu 18.01的系统上运行NNFormer，其中Python 3.6，Pytorch 1.8.1和CUDA 10.1。有关软件包和版本编号的完整列表，请参见Conda环境文件`environment.yml`. 

该软件利用图形处理单元（GPU）来加速神经网络培训和评估。因此，缺乏合适GPU的系统可能需要很长时间才能训练或评估模型。该软件已使用NVIDIA RTX 2080 TI GPU进行了测试，尽管我们预计其他GPU也将起作用，只要该单元提供足够的内存。

#### 2。安装指南
我们建议使用Conda软件包管理器安装所需的软件包，该软件包经理可通过Anaconda Python发行版获得。 Anaconda可以免费提供通过[Anaconda Inc](https://www.anaconda.com/products/individual)。安装Anaconda并克隆此存储库后，用作集成框架：
```
git clone https://github.com/282857341/nnFormer.git
cd nnFormer
conda env create -f environment.yml
source activate nnFormer
pip install -e .
```

#### 3。脚本和文件夹的功能
- **进行评估：**
  - ``nnFormer/nnformer/inference_acdc.py``
  
  - ``nnFormer/nnformer/inference_synapse.py``
  
  - ``nnFormer/nnformer/inference_tumor.py``
  
- **数据拆分：**
  - ``nnFormer/nnformer/dataset_json/``
  
- **用于推理：**
  - ``nnFormer/nnformer/inference/predict_simple.py``
  
- **网络体系结构：**
  - ``nnFormer/nnformer/network_architecture/nnFormer_acdc.py``
  
  - ``nnFormer/nnformer/network_architecture/nnFormer_synapse.py.py``
  
  - ``nnFormer/nnformer/network_architecture/nnFormer_tumor.py.py``
  
- **进行培训：**
  - ``nnFormer/nnformer/run/run_training.py``
  
- **数据集的培训师：**
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_acdc.py``
  
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_synapse.py.py``
  
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_tumor.py.py``
---

## 训练
#### 1。数据集下载
可以通过以下链接获取数据集：

**数据集i **
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**数据集II **
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**数据集III **
[Brain_tumor](http://medicaldecathlon.com/)

所有三个数据集的分割都可以在``nnFormer/nnformer/dataset_json/``.

#### 2。设置数据集
下载数据集后，您可以关注设置[nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md)用于路径配置和预处理过程。最后，您的文件夹应组织如下：

```
./Pretrained_weight/
./nnFormer/
./DATASET/
  ├── nnFormer_raw/
      ├── nnFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── nnFormer_cropped_data/
  ├── nnFormer_trained_models/
  ├── nnFormer_preprocessed/
```
您可以参考``nnFormer/nnformer/dataset_json/``对于数据拆分。

之后，您可以使用以下命令预处理上述数据：
```
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task01_ACDC
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task03_tumor

nnFormer_plan_and_preprocess -t 1
nnFormer_plan_and_preprocess -t 2
nnFormer_plan_and_preprocess -t 3
```

#### 3。培训和测试
- 培训和测试的命令：

```
bash train_inference.sh -c 0 -n nnformer_acdc -t 1 
#-c stands for the index of your cuda device
#-n denotes the suffix of the trainer located at nnFormer/nnformer/training/network_training/
#-t denotes the task index
```
如果您想使用自己的数据，请在路径中创建一个新的教练文件```nnformer/training/network_training```并确保教练文件中的类名与培训师文件相同。一些超参数可以在培训仪文件中进行调整，但是批处理大小和裁剪的大小应在文件中调整```nnformer/run/default_configuration.py```.
 
- 您可以通过此下载我们验证的模型权重[link](https://drive.google.com/drive/folders/1yvqlkeRq1qr5RxH-EzFyZEFsJsGFEc78?usp=sharing)。然后，您可以将模型权重及其关联文件放在相应的目录中。例如，在ACDC数据集上，它们应该像这样：
```
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model.pkl
```
#### 4。可视化结果

您可以从中下载NNFormer，nnunet和Unet的可视化结果[link](https://drive.google.com/file/d/1Lb4rIkwIpuJS3tomBiKl7FBtNF2dv_6M/view?usp=sharing).

#### 5。一个经常询问的问题
```
input feature has wrong size
```
如果您在实施过程中遇到此问题，请在``nnFormer/nnformer/run/default_configuration.py``我为每个数据集设置了独立的作物大小（即补丁大小）。您可能需要根据自己的需要来修改作物大小。
