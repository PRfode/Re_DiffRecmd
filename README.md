# Re: DiffRecmd
> 扩散推荐模型 DiffRec 论文结果的复现 \
> 论文查看：[\[2304.04971\] Diffusion Recommender Model](https://arxiv.org/abs/2304.04971) \
> 我们着重于三个模型：DiffRec/L-DiffRec/T-DiffRec在MovieLens-1M数据集上效果的复现

## Setup
使用 git 导入项目后，安装对应依赖。
  项目使用 `python==3.9`。

## Dataset Prepare
在原本的基础上，我们增添了Leave-One-Out的GDR数据集，以供给DiffRec的训练。
### MovieLens-1M 数据集
运行 `./data/MovieLens_1M/MovieLens-1M.ipynb` 下载并处理 MovieLens-1M 数据集。
### Grocery and Gourmet Food 数据集
运行 `./data/Grocery_and_Gourmet_Food/Amazon.ipynb` 下载并处理 Grocery and Gourmet Food 数据集。
## Model Run Example
你可以添加`--gpu 0`的字段来使用GPU进行训练（not guaranteed to work）。
```
python main.py --model_name DiffRec_c --dataset MovieLens_1M/ML_1MGDR --emb_size 10 --dims "[1000]" --lr 0.0001 --l2 0.0 --test_all 1 --steps 5 --noise_scale 0.01 --batch_size 400 --reweight 0 --noise_max=0.01 --noise_min=0.001  --early_stop 50 --main_metric HR@20
```

## Super Para Intro
- steps（训练步数）：在训练过程中，扩散模型的前向过程从时间步0到时间步T（即steps）。这个参数定义了训练时添加噪声的步骤数，即从原始数据到纯高斯噪声需要多少步。在训练时，我们会在0到steps-1之间随机采样时间步t，然后计算损失。
- sampling_steps（采样步数）：在推理（生成）过程中，我们从纯噪声开始，逐步去噪，直到生成最终的数据。这个参数定义了在推理时进行去噪的步骤数。sampling_steps可以不同于训练时的steps，但必须有sampling_steps不大于steps。我们可以使用更多的采样步骤来获得更好的生成质量，但也会增加计算成本。
- 噪声尺度（noise_scale）是全局噪声缩放因子，控制前向过程中添加噪声的强度。噪声边界参数控制噪声调度的下界和上界。噪声的强度随扩散过程中时间步变化而变化，其中noise_min决定初始噪声强度，noise_max决定最终噪声强度。
- 学习率（lr）：训练过程中模型参数的更新步长。
- 网络结构参数控制去噪网络MLP的架构，其中dims定义隐藏层维度，emb_size定义时间步嵌入维度。

## Model Intro
### DiffRec
~~DiffRec.py outdated~~

~~DiffRec_beta.py outdated~~

DiffRec_c.py current version

~~LDiffRec_a.py outdated~~

LDiffRec_b.py current version

~~TDiffRec_a.py outdated~~

TDiffRec_b.py current version

## Question that haven't been solved
1. 如果使用GPU进行训练，也就是在命令中添加 `--gpu 0` 的字段，那么则会出现GD模型参数被异常清零的情况，导致训练和测试异常。该异常经过debug定位到 BaseRunner.py 的 predict 方法的这一行：
```python
for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
```
我们曾使用合并模型和模块，注册参数到缓冲区，让GD模型的每个参数都正确保持在cuda上，更换更高版本的环境，但依然无法解决。

## Acknowledgments
This project uses code from [ReChorus](https://github.com/THUwangcy/ReChorus) which is released under the MIT License.
