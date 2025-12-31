# Re: DiffRecmd
> 扩散推荐模型 DiffRec 论文结果的复现 \
> 论文查看：[\[2304.04971\] Diffusion Recommender Model](https://arxiv.org/abs/2304.04971) \
> 我们着重于三个模型：DiffRec/L-DiffRec/T-DiffRec在MovieLens-1M(同时也有Grocery and Gourmet Food)数据集上效果的复现

## Setup
使用 git 导入项目后，安装对应依赖。

使用 `pip install -r requirements.txt` 或者是 `conda env create -f environment.yml` 安装依赖。

注意事项：
- 项目使用 `python==3.12.12`。
- 项目使用CUDA12.9版本，如果有需要请安装其他版本的torch。
- 项目仅保证在Windows/Mac环境下能够正确运行。

## Dataset Prepare
在原本的基础上，我们增添了Leave-One-Out的GDR数据集，以供给DiffRec的训练。
### MovieLens-1M 数据集
运行 `./data/MovieLens_1M/MovieLens-1M.ipynb` 下载并处理 MovieLens-1M 数据集。
### Grocery and Gourmet Food 数据集
运行 `./data/Grocery_and_Gourmet_Food/Amazon.ipynb` 下载并处理 Grocery and Gourmet Food 数据集。
## Model Run Example
你可以使用以下命令测试项目能否正常运行：
```powershell
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
```
你可以使用以下命令测试能否正常训练DiffRec模型：

```powershell
python main.py --model_name DiffRec_c --dataset MovieLens_1M/ML_1MGDR --emb_size 10 --dims "[1000]" --lr 0.0001 --l2 0.0 --test_all 1 --steps 5 --noise_scale 0.01 --batch_size 400 --reweight 0 --noise_max=0.01 --noise_min=0.001  --early_stop 50 --main_metric HR@20
```
你可以添加`--gpu 0`的字段来使用GPU进行训练（not guaranteed to work）。

## Model Intro
### DiffRec
~~DiffRec.py outdated~~

~~DiffRec_beta.py outdated~~

> DiffRec_c.py current version

~~LDiffRec_a.py outdated~~

> LDiffRec_b.py current version

~~TDiffRec_a.py outdated~~

> TDiffRec_b.py current version

## Super Param Intro
介绍由DiffRec、L-DiffRec、T-DiffRec三个模型新引入的超参数。
### DiffRec
- steps（训练步数）：训练阶段扩散过程从 t=0 到 t=T 共 steps 步。该值越大，前向加噪过程越细腻，但训练时间也随之增加。
- sampling_steps（采样步数）：推理阶段从纯噪声开始逆向去噪的步数。允许 sampling_steps ≤ steps；增大该值通常能提升生成质量，同时增加推理耗时。
- noise_scale（噪声幅度缩放）：把 noise_min 与 noise_max 线性拉伸的全局系数。最终每步实际使用的方差 = noise_scale * 从 noise_min 到 noise_max 的线性插值。
- noise_min（最小方差比）：决定第一步（t=0→1）添加噪声的相对强度下限。
- noise_max（最大方差比）：决定最后一步（t=T-1→T）添加噪声的相对强度上限。
- reweight（损失重加权开关）：取 0 时每一步损失权重相同；取 1 时根据信噪比 SNR 自动给不同 t 分配权重，以缓解低信噪步的梯度噪声问题。
- dims（DNN 隐层维度）：字符串形式传入，如 '[1000]' 或 '[512,256]'，决定去噪网络隐藏层宽度。
- emb_size（时间步嵌入维度）：把整数 t 映射成向量时使用的正弦嵌入维度，随后会与数据向量拼接进入 DNN。
- norm（输入归一化开关）：取 1 时在每一批数据进入 DNN 前做 L2 归一化。
### T-DiffRec
- w_min（最早交互权重）：用户行为序列按时间排序后，最早一条交互在训练向量里对应的初始权重，典型值 0.1。
- w_max（最新交互权重）：用户序列里最新一条交互对应的权重，典型值 1.0。

作用：把“时间衰减”显式写进输入向量，越新的行为权重越高，从而引导模型更关注近期兴趣，而无需修改网络结构。

### L-DiffRec
AutoEoncoder 超参数
- n_cate（物品隐类别数）：AutoEncoder 中间离散隐变量的类别数，用于学习物品聚类表示。
- in_dims（编码器各层宽度）：字符串形式，如 '[300]' 或 '[600,300]'，定义编码器隐藏层。
- out_dims（解码器各层宽度）：字符串形式，如 '[300,600]'，若留空 '[]' 则默认与编码器对称。
- act_func（AE 激活函数）：编码器/解码器内使用的非线性，可选 'tanh' 或 'relu'。
- reparam（变分开关）：取 1 时使用重参数化技巧，把编码器输出视为高斯分布，学习 VAE；取 0 退化为普通 AE。
- emb_path（预训练物品向量目录）：若目录下存在 dataset/item_emb.npy，则直接加载作为 AE 输入；否则随机初始化。

退火超参数
- lamda（扩散损失权重）：平衡 AE 重建损失与扩散 MSE 损失的系数。训练过程中会按 anneal_steps 退火至 anneal_cap。
- anneal_cap（lambda 退火下限）：lamda 最终不会小于该值，防止扩散项过早消失。
- anneal_steps（lambda 退火步数）：在前 N 步训练里线性降低 lamda，使 AE 先稳定，再逐步加入扩散约束。
- vae_anneal_cap（KL 退火上限）：若开启 reparam，控制 KL 项最大权重，防止早期 KL 塌陷。
- vae_anneal_steps（KL 退火步数）：在前 M 步训练里线性升高 KL 权重，让隐变量先学好，再逐步约束分布。

## Question that haven't been solved
1. 如果使用GPU进行训练，也就是在命令中添加 `--gpu 0` 的字段，那么则会出现GD模型参数被异常清零的情况，导致训练和测试异常。该异常经过debug定位到 BaseRunner.py 的 predict 方法的这一行：
```python
for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
```
就算我使用刚clone下来的ReChorus框架运行ReChorus自带的模型，也会出现这个问题。

这个问题仅在一台电脑上出现，仅在ReChorus项目中出现。

我们曾使用合并模型和模块，注册参数到缓冲区，让GD模型的每个参数都正确保持在cuda上，更换更高版本的环境，但依然无法解决。目前可以确定不是代码的问题。

## Acknowledgments
This project uses code from [ReChorus](https://github.com/THUwangcy/ReChorus) which is released under the MIT License.
