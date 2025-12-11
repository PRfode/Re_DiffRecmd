import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import enum
from models.BaseModel import BaseModel

# ========== 工具函数 ==========
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ========== DNN网络 ==========
class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        if isinstance(in_dims, int):
            in_dims = [in_dims]
            
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h

# ========== 模型均值类型枚举 ==========
class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

# ========== 主模型类 ==========
class GDModel(BaseModel):
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
        parser.add_argument('--beta_fixed', type=int, default=1, help='whether to fix beta_1')
        parser.add_argument('--noise_schedule', type=str, default='linear-var', 
                           choices=['linear', 'linear-var', 'cosine', 'binomial'])
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
        self.beta_fixed = bool(args.beta_fixed)
        self.noise_schedule = args.noise_schedule
        self.mean_type = ModelMeanType.START_X if args.mean_type == 'x0' else ModelMeanType.EPSILON

        # 去噪网络
        hidden = eval(args.dims)
        out_dims = hidden + [corpus.n_items]
        in_dims = [corpus.n_items] + hidden
        self.dnn = DNN(in_dims=in_dims,
                      out_dims=out_dims,
                      emb_size=args.emb_size,
                      time_type="cat",
                      norm=False,
                      dropout=0.5).to(self.device)

        # 重要性采样缓存
        self.history_num_per_term = 10
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))
        
        # 扩散过程相关参数
        if self.noise_scale != 0.:
            betas = torch.tensor(self._get_betas(), dtype=torch.float64).to(self.device)
            
            if self.beta_fixed:
                betas[0] = 0.00001
                
            self.register_buffer('betas', betas)
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self._calculate_diffusion_params()
        
        self.test_all = 1

    # ========== 扩散过程相关方法 ==========
    def _get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def _calculate_diffusion_params(self):
        """Calculate all diffusion-related parameters and register them as buffers."""
        alphas = 1.0 - self.betas
        
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), alphas_cumprod[:-1]])
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)])
        
        # 注册所有扩散过程相关参数
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # 计算后验分布参数
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])
        )
        posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        # 注册后验分布参数
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        sqrt_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        """Reverse diffusion process: p(x_{t-1} | x_t)"""
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def p_mean_variance(self, model, x, t):
        """Compute mean and variance of p(x_{t-1} | x_t)"""
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute mean and variance of q(x_{t-1} | x_t, x_0)"""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        """Predict x_0 from epsilon prediction"""
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D tensor for a batch of indices.
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    # ========== 训练方法 ==========
    def forward(self, feed_dict):
        x_start = feed_dict['vector'].float()          # (B, n_item)  0-1
        batch_size = x_start.size(0)

        # 采样 t
        if self.reweight and (self.Lt_count == self.history_num_per_term).all():
            Lt_sqrt = torch.sqrt((self.Lt_history ** 2).mean(-1))
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            pt_all = pt_all * 0.999 + 0.001 / self.steps
            t = torch.multinomial(pt_all, batch_size, replacement=True)
            pt = pt_all[t] * self.steps
        else:
            t = torch.randint(0, self.steps, (batch_size,), device=self.device)
            pt = torch.ones_like(t).float()

        # 前向加噪
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # 去噪网络预测
        pred = self.dnn(x_t, t)

        # 损失
        target = {'start_x': x_start, 'epsilon': noise}[self.mean_type.name.lower()]
        mse = (pred - target).pow(2).mean(-1)          # (B,)
        if self.reweight:
            weight = (self.alphas_cumprod[t-1] - self.alphas_cumprod[t]).clamp(min=0)
        else:
            weight = torch.ones_like(mse)
        loss = (weight * mse / pt).mean()

        # 更新重要性缓冲
        with torch.no_grad():
            for ti, loss_val in zip(t, mse):
                if self.Lt_count[ti] == self.history_num_per_term:
                    self.Lt_history[ti, :-1] = self.Lt_history[ti, 1:].clone()
                    self.Lt_history[ti, -1] = loss_val
                else:
                    self.Lt_history[ti, self.Lt_count[ti]] = loss_val
                    self.Lt_count[ti] += 1

        return {'loss': loss, 'prediction': torch.zeros(batch_size, 1, device=self.device)}

    def loss(self, out_dict):
        return out_dict['loss']

    # ========== 推理方法 ==========
    def inference(self, feed_dict):
        user_ids = feed_dict['user_id']
        batch_size = len(user_ids)
        x = torch.randn(batch_size, self.corpus.n_items, device=self.device)
        steps = self.sampling_steps

        with torch.no_grad():
            x = self.p_sample(
                model=lambda xt, t: self.dnn(xt, t),
                x_start=x,
                steps=steps,
                sampling_noise=True
            )
        return {'prediction': x}   # (B, n_item) 评分向量
    
    def eval(self):
        super().eval()
        self.dnn.eval()

    # ========== Dataset类 ==========
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
                    'item_id': np.array([0])}  # 占位

        def collate_batch(self, feed_dicts):
            batch = {}
            batch['user_id'] = torch.tensor([d['user_id'] for d in feed_dicts])
            batch['vector'] = torch.from_numpy(np.stack([d['vector'] for d in feed_dicts]))
            batch['item_id'] = torch.from_numpy(np.array([d['item_id'] for d in feed_dicts]))
            return batch