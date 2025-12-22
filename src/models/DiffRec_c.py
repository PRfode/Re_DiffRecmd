# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.BaseModel import BaseModel
from collections import defaultdict

# ============================================================================
# 辅助函数
# ============================================================================
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# ============================================================================
# 核心模型：DiffRec_c
# ============================================================================
class DiffRec_c(BaseModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['steps', 'noise_scale', 'reweight']

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

        # # 【Magic Fix 1】: 清空 residual_clicked_set
        # # BaseRunner 默认会Mask掉这里面的物品。但在 Leave-One-Out 模式下，
        # # 这里面存的恰恰是我们要预测的 Target (Dev/Test Item)。
        # # 所以必须清空它，防止正确答案被设为 -inf。
        # self.corpus.residual_clicked_set = defaultdict(set)

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

        self.dnn = DNN(corpus.n_items, self.dims, self.emb_size, self.norm)
        self._build_diffusion_params()
        
        self.history_num_per_term = 10
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))

    def _build_diffusion_params(self):
        if self.noise_scale == 0:
            self.betas = torch.tensor([0.0] * self.steps).float().to(self.device)
        else:
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            map_variance = np.linspace(start, end, self.steps, dtype=np.float64)
            betas = betas_from_linear_variance(self.steps, map_variance)
            betas[0] = 0.00001
            self.register_buffer('betas', torch.tensor(betas).float())
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), alphas_cumprod[:-1]])
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(
            torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])
        ))
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

    def SNR(self, t):
        t = torch.clamp(t, min=0)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

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
        mse = mean_flat((x_start - predicted_x_start) ** 2)
        
        if self.reweight:
            snr = self.alphas_cumprod[ts] / (1 - self.alphas_cumprod[ts])
            snr_prev = self.alphas_cumprod_prev[ts] / (1 - self.alphas_cumprod_prev[ts])
            weight = torch.where((ts == 0), torch.tensor(1.0, device=self.device), snr_prev - snr)
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

    # ========================================================================== #
    #                   code from gaussian_diffusion.py                          #
    # ========================================================================== #

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # 反向去噪
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            # print('p_sample come to q_sample')
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
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        # if self.mean_type == ModelMeanType.START_X:
        #     pred_xstart = model_output
        # elif self.mean_type == ModelMeanType.EPSILON:
        #     pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        # else:
        #     raise NotImplementedError(self.mean_type)
        pred_xstart = model_output
        
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

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def see_betas(self):
        print(f"betas: {self.betas}")
        print(f"sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod}")

    def betas_from_linear_variance(steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def normal_kl(mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    # ========================================================================== #
    #                        gaussian_diffusion code end                         #
    # ========================================================================== #

    def inference(self, feed_dict):
        batch_users = feed_dict['user_id']
        batch_size = len(batch_users)
        
        # x = torch.randn(batch_size, self.corpus.n_items).to(self.device)
        x_start = feed_dict['vector'].float()
        with torch.no_grad():
            # 关键修改：使用用户历史向量作为条件
            x = self.p_sample(
                model=lambda xt, t: self.dnn(xt, t),
                x_start=x_start,
                steps=self.sampling_steps,
                sampling_noise=True
            ).to(self.device)
        
        indices = list(range(self.sampling_steps))[::-1]
        
        with torch.no_grad():
            for i in indices:
                t = torch.tensor([i] * batch_size).to(self.device)
                pred_x0 = self.dnn(x, t)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) 
                
                model_mean = (
                    self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_x0 +
                    self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
                )
                
                if i > 0:
                    noise = torch.randn_like(x)
                    model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                    x = model_mean + torch.exp(0.5 * model_log_variance) * noise
                else:
                    x = model_mean

        # 【Magic Fix 2】: Swap Trick (列交换)
        
        # 1. 获取目标 Item ID，并强制展平为 [Batch_Size]
        # 使用 .view(-1) 确保它是 1D 的，解决 RuntimeError
        target_items = feed_dict['item_id'].view(-1) 
        
        # 2. 取出第 0 列的分数 (备份)
        col0_scores = x[:, 0].clone()
        
        # 3. 取出目标列的分数
        batch_indices = torch.arange(batch_size).to(self.device)
        target_scores = x[batch_indices, target_items]
        
        # 4. 交换
        # 现在 target_scores 是 [Batch_Size]，x[:, 0] 也是 [Batch_Size]，形状匹配了
        x[:, 0] = target_scores
        x[batch_indices, target_items] = col0_scores

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
            
            # x0: 历史交互
            clicked_items = list(self.corpus.train_clicked_set.get(user_id, []))
            vector = np.zeros(self.corpus.n_items, dtype=np.float32)
            vector[clicked_items] = 1.0
            
            # target: 真实的目标物品 ID
            # 在 Train 阶段没有单一 target，返回 0
            # 在 Dev/Test 阶段，BaseReader 的 data_df 里存了 target item_id
            target_item = 0
            if self.phase != 'train':
                # 获取该用户在 Dev/Test DataFrame 中的 item_id
                # 注意：self.corpus.data_df[self.phase] 每一行是一个 (user, item)
                # 我们这里简化处理，假设每个用户只有一条 Dev/Test 数据 (Leave-One-Out)
                # 这种查找效率较低，但对于 ML-1M 可接受。优化版应该预处理成 dict。
                # ReChorus 的 BaseReader 其实本身就支持按行读取，但我们为了 DiffRec 这种 User-based Batch 改写了。
                # 快速查找法：
                user_rows = self.corpus.data_df[self.phase]
                target_item = user_rows[user_rows['user_id'] == user_id]['item_id'].values[0]

            return {
                'user_id': user_id,
                'vector': vector,
                'item_id': np.array([target_item]) 
            }
        
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
        self.init_weights()
        
    def init_weights(self):
        for layer in self.layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.time_emb:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                std = np.sqrt(2.0 / (size[0] + size[1]))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = self._timestep_embedding(timesteps, self.emb_size).to(x.device)
        time_emb = self.time_emb(time_emb)
        if self.norm: x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, time_emb], dim=-1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = torch.tanh(h)
        return h

    def _timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding