# Re: DiffRecmd
> 扩散推荐模型 DiffRec 的复现 \
> 论文查看：[\[2304.04971\] Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)

---

## Setup
使用 git 导入项目后，安装对应依赖。

若使用PyCharm，请将src文件夹设置为Sources Root以避免包导入错误。

---

## Dataset Prepare
DiffRec 是给出用户的下一步点击概率，所以需要准备topK即可
### MovieLens-1M 数据集
运行 `src/data/MovieLens_1M/MovieLens-1M.ipynb` 下载并处理 MovieLens-1M 数据集。

---

## Train Record
我们使用MovieLens-1M数据集进行训练以对论文进行复现，训练脚本为 `src/main.py`。

使用以下参数：
```
'--model_name' 'DiffRec_c' '--dataset' 'MovieLens_1M/ML_1MGDR' '--emb_size' '10' '--dims' '[1000]' '--lr' '0.0001' '--l2' '0.0' '--test_all' '1' '--steps' '5' '--noise_scale' '0.01' '--noise_min' '0.001' '--noise_max' '0.01' '--main_metric' 'HR@20' '--batch_size' '400' '--reweight' '0' '--early_stop' '50' 
```

在 2025-12-11 17:12:42 的训练中，
- 在 epoch 88，第一次实现在验证集上 HR@20 突破 5%，达到 0.0514。

进一步优化模型后，在 2025-12-12 23:13:08 的训练中，
- 在 epoch 25，第一次实现在验证集上 HR@20 突破 10%，达到 0.1024。
- 在 epoch 142，效果最优，在验证集上 HR@20 达到 0.1452。

---
## Question that haven't been solved
1. 如果使用GPU进行训练，也就是在命令中添加 `--gpu 0` 的字段，那么则会出现GD模型参数被异常清零的情况，导致训练和测试异常。该异常经过debug定位到 BaseRunner.py 的 predict 方法的这一行：
```python
for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
```
我们曾使用合并模型和模块，注册参数到缓冲区，让GD模型的每个参数都正确保持在cuda上，更换更高版本的环境，但依然无法解决。

## Acknowledgments
This project uses code from [ReChorus](https://github.com/THUwangcy/ReChorus) which is released under the MIT License.