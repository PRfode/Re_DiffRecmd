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

## Acknowledgments
This project uses code from [ReChorus](https://github.com/THUwangcy/ReChorus) which is released under the MIT License.