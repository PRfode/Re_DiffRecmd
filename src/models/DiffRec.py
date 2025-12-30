# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.BaseModel import BaseModel

class DiffRec(BaseModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['steps', 'noise_scale', 'noise_min', 'noise_max']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--dims', type=str, default='[1000]', help='The dims for the DNN.')
        parser.add_argument('--norm', type=int, default=0, help='Normalize the input or not.')
        parser.add_argument('--emb_size', type=int, default=10, help='Timestep embedding size.')
        parser.add_argument('--steps', type=int, default=100, help='Diffusion steps.')
        parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale.')
        parser.add_argument('--noise_min', type=float, default=0.0001, help='Noise lower bound.')
        parser.add_argument('--noise_max', type=float, default=0.02, help='Noise upper bound.')
        parser.add_argument('--sampling_steps', type=int, default=0, help='Steps during inference (0 means = steps).')
        parser.add_argument('--reweight', type=int, default=1, help='Assign different weight to different timestep.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.corpus = corpus

        self.dims = eval(args.dims)
        self.norm = bool(args.norm)
        self.emb_size = args.emb_size
        self.steps = args.steps
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.sampling_steps = args.sampling_steps if args.sampling_steps > 0 else self.steps
        self.reweight = bool(args.reweight)
        
        # 强制设置 test_all 为 1
        # DiffRec 一次性生成所有物品的评分
        self.test_all = 1 

        self.dnn = DNN(corpus.n_items, self.dims, self.emb_size, self.norm)

        self._build_diffusion_params()

        # 重要性采样状态
        self.history_num_per_term = 10
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))
        
        self.apply(self.init_weights)

    def _build_diffusion_params(self):
        """预计算扩散过程所需的 alphas, betas 等参数"""
        if self.noise_scale == 0:
            self.betas = torch.tensor([0.0] * self.steps).float().to(self.device)
        else:
            # Linear schedule
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            betas = np.linspace(start, end, self.steps, dtype=np.float64)
            # Beta trick (from original paper)
            betas[0] = 0.00001
            self.register_buffer('betas', torch.tensor(betas).float())
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), alphas_cumprod[:-1]])
        self.register_buffer('posterior_variance', self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _sample_timesteps(self, batch_size):
        """重要性采样时间步 t"""
        if self.reweight:
            if not (self.Lt_count == self.history_num_per_term).all():
                return torch.randint(0, self.steps, (batch_size,), device=self.device).long(), torch.ones(batch_size, device=self.device)
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, dim=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all = pt_all * (1 - 0.001) + 0.001 / len(pt_all) # 平滑
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all[t] * len(pt_all)
            return t, pt
        else:
            return torch.randint(0, self.steps, (batch_size,), device=self.device).long(), torch.ones(batch_size, device=self.device)

    def forward(self, feed_dict):
        """
        训练过程：
        1. 获取真实交互向量 x_0
        2. 采样时间步 t
        3. 加噪得到 x_t
        4. DNN 预测 x_0
        5. 计算 Loss 并返回
        """
        x_start = feed_dict['vector'].float() # [batch_size, n_items]
        batch_size = x_start.shape[0]
        
        # 采样时间步
        ts, pt = self._sample_timesteps(batch_size)
        
        # 加噪 q_sample
        noise = torch.randn_like(x_start)
        x_t = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, ts, x_start.shape) * x_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, ts, x_start.shape) * noise
        )
        
        # 模型预测
        predicted_x_start = self.dnn(x_t, ts)
        
        # 计算 MSE Loss
        mse = torch.mean((x_start - predicted_x_start) ** 2, dim=-1) # [batch_size]
        
        # Reweight Loss
        if self.reweight:
            # SNR weight (mean_type == x0)
            snr = self.alphas_cumprod[ts] / (1 - self.alphas_cumprod[ts])
            snr_prev = self.alphas_cumprod[ts-1] / (1 - self.alphas_cumprod[ts-1])
            # Handle t=0
            weight = torch.where(ts == 0, torch.tensor(1.0, device=self.device), snr_prev - snr)
            weight = torch.clamp(weight, min=0) # 防止数值问题
        else:
            weight = torch.ones_like(mse)
            
        losses = weight * mse
        
        # 更新 Importance Sampling 历史
        with torch.no_grad():
            for t, loss_val in zip(ts, losses):
                if self.Lt_count[t] == self.history_num_per_term:
                    self.Lt_history[t] = torch.cat([self.Lt_history[t, 1:], loss_val.unsqueeze(0)])
                else:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val
                    self.Lt_count[t] += 1

        # 返回 Loss
        final_loss = torch.mean(losses / pt)
        return {'loss': final_loss,
            'prediction': torch.zeros(batch_size, 1, device=self.device)
                }

    def loss(self, out_dict):
        return out_dict['loss']

    def inference(self, feed_dict):
        """
        推断过程 (Generate)：
        1. 从纯噪声 x_T 开始
        2. 逐步去噪 t -> 0
        3. 返回预测的 x_0
        """
        # 只需要 batch_size 信息，或者用来 mask history
        batch_users = feed_dict['user_id']
        batch_size = len(batch_users)
        
        # 从纯噪声开始
        x = torch.randn(batch_size, self.corpus.n_items).to(self.device)
        
        # 逐步去噪
        indices = list(range(self.sampling_steps))[::-1]
        
        with torch.no_grad():
            for i in indices:
                t = torch.tensor([i] * batch_size).to(self.device)
                
                # 模型预测 x_0
                pred_x0 = self.dnn(x, t)
                
                # 计算后验分布均值 (p_mean_variance)
                model_mean = (
                    self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_x0 +
                    self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
                )
                
                if i > 0:
                    noise = torch.randn_like(x)
                    # p_sample 时的方差项
                    model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
                    # log_variance clipped (避免 log(0))
                    # 这里简化处理，直接用 variance
                    x = model_mean + torch.sqrt(model_variance) * noise
                else:
                    x = model_mean

        return {'prediction': x}

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """辅助函数：根据 t 索引提取 tensor 并广播"""
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    class Dataset(BaseModel.Dataset):
        def __init__(self, model, corpus, phase):
            self.model = model
            self.corpus = corpus
            self.phase = phase
            self.buffer_dict = dict()
            
            # DiffRec 的 Dataset 逻辑：
            # 训练集：需要遍历所有 User。
            # 测试集：BaseRunner 默认也是按 Batch 遍历，但 DiffRec 需要构建历史交互向量。
            
            if phase == 'train':
                users = list(corpus.train_clicked_set.keys())
            else:
                users = corpus.data_df[phase]['user_id'].unique().tolist()

            self.data = {'user_id': users}

        def __len__(self):
            return len(self.data['user_id'])

        def _get_feed_dict(self, index):
            user_id = self.data['user_id'][index]
            clicked_items = list(self.corpus.train_clicked_set.get(user_id, []))

            vector = np.zeros(self.corpus.n_items, dtype=np.float32)
            vector[clicked_items] = 1.0
            
            feed_dict = {
                'user_id': user_id,
                'vector': vector,
                'item_id': np.array([0]) # 占位符，兼容 BaseRunner
            }
            return feed_dict
        
        def collate_batch(self, feed_dicts):
            feed_dict = {}
            feed_dict['user_id'] = torch.tensor([d['user_id'] for d in feed_dicts])
            vectors = np.array([d['vector'] for d in feed_dicts])
            feed_dict['vector'] = torch.from_numpy(vectors)

            item_ids = np.array([d['item_id'] for d in feed_dicts])
            feed_dict['item_id'] = torch.from_numpy(item_ids)

            return feed_dict

class DNN(nn.Module):
    def __init__(self, in_dims, hidden_dims, emb_size, norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.emb_size = emb_size
        self.norm = norm
        
        # Time Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
        )
        
        # Input: Item_num + Time_Emb
        dims = [in_dims + emb_size] + hidden_dims + [in_dims]
        
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, timesteps):
        # 1. 计算 Time Embedding
        time_emb = self._timestep_embedding(timesteps, self.emb_size).to(x.device)
        time_emb = self.time_emb(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # 2. Concat
        h = torch.cat([x, time_emb], dim=-1)
        
        # 3. MLP Forward
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = torch.tanh(h) # 论文使用 Tanh
                
        return h

    def _timestep_embedding(self, timesteps, dim, max_period=10000):
        """Sinusoidal embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding