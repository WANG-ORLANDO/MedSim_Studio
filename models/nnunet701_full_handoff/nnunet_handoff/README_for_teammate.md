# nnU-Net v2 TotalSegOrgans6 模型交接说明

## 1. 模型信息

- Framework: nnU-Net v2
- Dataset: Dataset701_TotalSegOrgans6
- Configuration: 3d_fullres
- Fold: 0
- Checkpoint: checkpoint_best.pth / checkpoint_final.pth

## 2. 标签定义

0 background  
1 liver  
2 kidney  
3 lung  
4 spleen  
5 pancreas  
6 bladder  

## 3. 当前验证结果

Validation foreground mean Dice: 0.9178

Per-class Dice:

- liver: 0.9425
- kidney: 0.8968
- lung: 0.9672
- spleen: 0.9323
- pancreas: 0.8611
- bladder: 0.9071

注意：这是 nnU-Net fold 0 内部验证集结果，不是外部测试集结果。

## 4. 目录说明

model/Dataset701_TotalSegOrgans6/
是完整 nnU-Net v2 results 目录，包含模型权重、plans、训练日志和验证结果。

visual_cases/
包含若干 CT 原图和预测 mask，可直接用 3D Slicer 或 ITK-SNAP 打开查看效果。

summary_fold0.json
是 fold 0 validation 的完整指标文件。

## 5. 本地推理准备

搭档本地需要安装 nnU-Net v2，并设置环境变量：

export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

然后把：

model/Dataset701_TotalSegOrgans6

复制或移动到：

$nnUNet_results/Dataset701_TotalSegOrgans6

## 6. 输入格式

待推理 CT 文件必须命名为：

caseid_0000.nii.gz

例如：

input/
  test001_0000.nii.gz

## 7. 推理命令示例

nnUNetv2_predict \
  -i input \
  -o output \
  -d 701 \
  -c 3d_fullres \
  -f 0 \
  -chk checkpoint_best.pth

输出示例：

output/test001.nii.gz

输出 mask 的标签值为 0-6，对应上面的器官类别。
