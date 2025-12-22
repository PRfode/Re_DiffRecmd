# Re: DiffRecmd
> 扩散推荐模型 DiffRec 论文结果的复现 \
> 论文查看：[\[2304.04971\] Diffusion Recommender Model](https://arxiv.org/abs/2304.04971) \
> 我们着重于四个模型：DiffRec/L-DiffRec/T-DiffRec/LT-DiffRec在MovieLens-1M数据集上效果的复现

## Setup
使用 git 导入项目后，安装对应依赖。
  项目使用 `python==3.9`。

若使用PyCharm，请将src文件夹设置为Sources Root以避免包导入错误。

## Dataset Prepare
DiffRec 是给出用户的下一步点击概率，所以需要准备topK即可
### MovieLens-1M 数据集
运行 `src/data/MovieLens_1M/MovieLens-1M.ipynb` 下载并处理 MovieLens-1M 数据集。


## Super Para Intro
- steps（训练步数）：在训练过程中，扩散模型的前向过程从时间步0到时间步T（即steps）。这个参数定义了训练时添加噪声的步骤数，即从原始数据到纯高斯噪声需要多少步。在训练时，我们会在0到steps-1之间随机采样时间步t，然后计算损失。
- sampling_steps（采样步数）：在推理（生成）过程中，我们从纯噪声开始，逐步去噪，直到生成最终的数据。这个参数定义了在推理时进行去噪的步骤数。sampling_steps可以不同于训练时的steps，但必须有sampling_steps不大于steps。我们可以使用更多的采样步骤来获得更好的生成质量，但也会增加计算成本。
- 

## Train Record
### DiffRec
我们使用MovieLens-1M数据集进行训练以对论文进行复现，训练脚本为 `src/main.py`，参数设置见底部。

在 2025-12-11 17:12:42 的训练中，
- 在 epoch 88，第一次实现在验证集上 HR@20 突破 5%，达到 0.0514。

进一步优化模型后，在 2025-12-12 23:13:08 的训练中，
- 在 epoch 25，第一次实现在验证集上 HR@20 突破 10%，达到 0.1024。
- 在 epoch 142，效果最优，在验证集上 HR@20 达到 0.1452（ideal）。

## Question that haven't been solved
1. 如果使用GPU进行训练，也就是在命令中添加 `--gpu 0` 的字段，那么则会出现GD模型参数被异常清零的情况，导致训练和测试异常。该异常经过debug定位到 BaseRunner.py 的 predict 方法的这一行：
```python
for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
```
我们曾使用合并模型和模块，注册参数到缓冲区，让GD模型的每个参数都正确保持在cuda上，更换更高版本的环境，但依然无法解决。

## Acknowledgments
This project uses code from [ReChorus](https://github.com/THUwangcy/ReChorus) which is released under the MIT License.
