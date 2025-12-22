import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.BaseModel import BaseModel

# --- 辅助函数：正确的 Linear Variance 调度计算 ---
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

class DiffRec_beta(BaseModel):
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
        
        self.test_all = 1 

        # 构建 DNN
        self.dnn = DNN(corpus.n_items, self.dims, self.emb_size, self.norm)
        
        # 构建 Diffusion 参数
        self._build_diffusion_params()
        
        # 重要性采样状态维护
        self.history_num_per_term = 10
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))
        
        self.apply(self.init_weights)

    def _build_diffusion_params(self):
        """【关键修复】预计算扩散过程参数，使用 Linear Variance Schedule"""
        if self.noise_scale == 0:
            self.betas = torch.tensor([0.0] * self.steps).float().to(self.device)
        else:
            # Linear-var schedule
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            
            # 1. 先生成线性的方差序列
            map_variance = np.linspace(start, end, self.steps, dtype=np.float64)
            # 2. 再将方差转换为 Beta
            betas = betas_from_linear_variance(self.steps, map_variance)
            
            # Beta fixed trick (from original paper)
            betas[0] = 0.00001
            self.register_buffer('betas', torch.tensor(betas).float())
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 计算后验分布参数 (用于推断)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), alphas_cumprod[:-1]])
        self.register_buffer('posterior_variance', self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _sample_timesteps(self, batch_size):
        if self.reweight:
            if not (self.Lt_count == self.history_num_per_term).all():
                return torch.randint(0, self.steps, (batch_size,), device=self.device).long(), torch.ones(batch_size, device=self.device)
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, dim=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all = pt_all * (1 - 0.001) + 0.001 / len(pt_all)
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all[t] * len(pt_all)
            return t, pt
        else:
            return torch.randint(0, self.steps, (batch_size,), device=self.device).long(), torch.ones(batch_size, device=self.device)

    def forward(self, feed_dict):
        x_start = feed_dict['vector'].float()
        batch_size = x_start.shape[0]
        
        ts, pt = self._sample_timesteps(batch_size)
        
        noise = torch.randn_like(x_start)
        x_t = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, ts, x_start.shape) * x_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, ts, x_start.shape) * noise
        )
        
        predicted_x_start = self.dnn(x_t, ts)
        
        mse = torch.mean((x_start - predicted_x_start) ** 2, dim=-1)
        
        if self.reweight:
            snr = self.alphas_cumprod[ts] / (1 - self.alphas_cumprod[ts])
            snr_prev = self.alphas_cumprod[ts-1] / (1 - self.alphas_cumprod[ts-1])
            weight = torch.where(ts == 0, torch.tensor(1.0, device=self.device), snr_prev - snr)
            weight = torch.clamp(weight, min=0)
        else:
            weight = torch.ones_like(mse)
            
        losses = weight * mse
        
        with torch.no_grad():
            for t, loss_val in zip(ts, losses):
                if self.Lt_count[t] == self.history_num_per_term:
                    self.Lt_history[t] = torch.cat([self.Lt_history[t, 1:], loss_val.unsqueeze(0)])
                else:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val
                    self.Lt_count[t] += 1

        final_loss = torch.mean(losses / pt)
        return {'loss': final_loss, 'prediction': torch.zeros(batch_size, 1, device=self.device)}

    def loss(self, out_dict):
        return out_dict['loss']

    def inference(self, feed_dict):
        batch_users = feed_dict['user_id']
        batch_size = len(batch_users)
        
        x = torch.randn(batch_size, self.corpus.n_items).to(self.device)
        indices = list(range(self.sampling_steps))[::-1]
        
        with torch.no_grad():
            for i in indices:
                t = torch.tensor([i] * batch_size).to(self.device)
                pred_x0 = self.dnn(x, t)
                
                # 【关键优化】增加 Clamping，防止预测值发散，提升稳定性
                # 训练目标是 0/1，所以限制在 [0, 1] 或 [-1, 1] 之间是合理的
                # 原论文使用 x0-prediction, 这里稍微放宽一点范围防止梯度截断太死
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                model_mean = (
                    self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_x0 +
                    self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
                )
                
                if i > 0:
                    noise = torch.randn_like(x)
                    model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
                    x = model_mean + torch.sqrt(model_variance) * noise
                else:
                    x = model_mean

        return {'prediction': x}

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
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
                'item_id': np.array([0]) 
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
        
        self.time_emb = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
        )
        
        dims = [in_dims + emb_size] + hidden_dims + [in_dims]
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, timesteps):
        time_emb = self._timestep_embedding(timesteps, self.emb_size).to(x.device)
        time_emb = self.time_emb(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        h = torch.cat([x, time_emb], dim=-1)
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = torch.tanh(h)
                
        return h

    def _timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding