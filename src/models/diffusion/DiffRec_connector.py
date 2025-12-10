# -*- coding: utf-8 -*-
"""
直接把现成的 DNN + GaussianDiffusion 嵌进 ReChorus
无需改动原 DNN.py / gaussian_diffusion.py
"""
import torch
import torch.nn as nn
import numpy as np
from models.BaseModel import BaseModel
# -------------- 现成模块 -----------------
from models.diffusion.DNN import DNN                    # 你的 DNN.py
from models.diffusion.gaussian_diffusion import (      # 你的 gaussian_diffusion.py
    GaussianDiffusion, ModelMeanType
)


class DiffRec_connector(BaseModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['steps', 'noise_scale', 'noise_min', 'noise_max']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--dims', type=str, default='[1000]', help='MLP hidden dims')
        parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding dim')
        parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
        parser.add_argument('--noise_scale', type=float, default=0.1)
        parser.add_argument('--noise_min', type=float, default=0.0001)
        parser.add_argument('--noise_max', type=float, default=0.02)
        parser.add_argument('--sampling_steps', type=int, default=0, help='inference steps; 0→=steps')
        parser.add_argument('--mean_type', type=str, default='x0', choices=['x0', 'eps'])
        parser.add_argument('--reweight', type=int, default=1, help='loss re-weight')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.corpus = corpus
        self.steps = args.steps
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.sampling_steps = args.sampling_steps if args.sampling_steps > 0 else args.steps
        self.reweight = bool(args.reweight)
        self.mean_type = ModelMeanType.START_X if args.mean_type == 'x0' else ModelMeanType.EPSILON

        # 1. 去噪网络
        hidden = eval(args.dims)
        out_dims = hidden + [corpus.n_items]
        in_dims = [corpus.n_items] + hidden
        self.dnn = DNN(in_dims=in_dims,
                       out_dims=out_dims,
                       emb_size=args.emb_size,
                       time_type="cat",
                       norm=False,
                       dropout=0.5).to(self.device)

        # 2. 扩散过程封装
        self.diffusion = GaussianDiffusion(
            mean_type=self.mean_type,
            noise_schedule='linear-var',
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
            steps=self.steps,
            device=self.device,
            history_num_per_term=10
        )

        # # 3. 重要性采样缓存
        # self.history_num_per_term = 10
        # self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term))
        # self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))
        # self.test_all = 1  # 评测时用全物品池

    # ---------- 训练 ----------
    def forward(self, feed_dict):
        x_start = feed_dict['vector'].float()          # (B, n_item)  0-1
        batch_size = x_start.size(0)

        # 采样 t
        if self.reweight and (self.diffusion.Lt_count == self.diffusion.history_num_per_term).all():
            Lt_sqrt = torch.sqrt((self.diffusion.Lt_history ** 2).mean(-1))
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            pt_all = pt_all * 0.999 + 0.001 / self.steps
            t = torch.multinomial(pt_all, batch_size, replacement=True)
            pt = pt_all[t] * self.steps
        else:
            t = torch.randint(0, self.steps, (batch_size,), device=self.device)
            pt = torch.ones_like(t).float()

        # 前向加噪
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise)

        # 去噪网络预测
        pred = self.dnn(x_t, t)        # 与目标形状相同

        # 损失
        target = {'start_x': x_start, 'epsilon': noise}[self.mean_type.name.lower()]
        mse = (pred - target).pow(2).mean(-1)          # (B,)
        if self.reweight:
            weight = (self.diffusion.alphas_cumprod[t-1] - self.diffusion.alphas_cumprod[t]).clamp(min=0)
        else:
            weight = torch.ones_like(mse)
        loss = (weight * mse / pt).mean()

        # 更新重要性缓冲
        with torch.no_grad():
            for ti, loss_val in zip(t, mse):
                if self.diffusion.Lt_count[ti] == self.history_num_per_term:
                    self.diffusion.Lt_history[ti, :-1] = self.diffusion.Lt_history[ti, 1:].clone()
                    self.diffusion.Lt_history[ti, -1] = loss_val
                else:
                    self.diffusion.Lt_history[ti, self.diffusion.Lt_count[ti]] = loss_val
                    self.diffusion.Lt_count[ti] += 1

        return {'loss': loss, 'prediction': torch.zeros(batch_size, 1, device=self.device)}

    def loss(self, out_dict):
        return out_dict['loss']

    # ---------- 推理 ----------
    def inference(self, feed_dict):
        user_ids = feed_dict['user_id']
        batch_size = len(user_ids)
        x = torch.randn(batch_size, self.corpus.n_items, device=self.device)
        steps = self.sampling_steps

        with torch.no_grad():
            x = self.diffusion.p_sample(
                model=lambda xt, t: self.dnn(xt, t),
                x_start=x,
                steps=steps,
                sampling_noise=True
            )
        print(f"Prediction shape: {x.shape}")  # 打印预测向量的形状
        print(f"Prediction values: {x[:5]}")  # 打印前5个预测值
        return {'prediction': x}   # (B, n_item) 评分向量
    
    def eval(self):
        print("overriding Eval mode")
        super().eval()
        self.diffusion.eval()
        self.dnn.eval()

    # ---------- Dataset ----------
    class Dataset(BaseModel.Dataset):
        def __init__(self, model, corpus, phase):
            self.model = model
            self.corpus = corpus
            self.phase = phase
            
            self.buffer_dict = {}
            
            # 用户列表
            if phase == 'train':
                users = list(corpus.train_clicked_set.keys())
            else:
                users = corpus.data_df[phase]['user_id'].unique().tolist()
            self.data = {'user_id': users}

        def __len__(self):
            return len(self.data['user_id'])

        def _get_feed_dict(self, index):
            user_id = self.data['user_id'][index]
            clicked = list(self.corpus.train_clicked_set.get(user_id, []))
            vector = np.zeros(self.corpus.n_items, dtype=np.float32)
            vector[clicked] = 1.0
            return {'user_id': user_id, 'vector': vector,
                    'item_id': np.array([0])}  # 占位，runner 会重填候选集

        def collate_batch(self, feed_dicts):
            batch = {}
            batch['user_id'] = torch.tensor([d['user_id'] for d in feed_dicts])
            batch['vector'] = torch.from_numpy(np.stack([d['vector'] for d in feed_dicts]))
            batch['item_id'] = torch.from_numpy(np.array([d['item_id'] for d in feed_dicts]))
            return batch
        
        # def eval(self):
        #     super().eval()
        #     self.model.diffusion.eval()
        #     self.model.dnn.eval()
