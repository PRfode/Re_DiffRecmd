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
# AutoEncoder 模型
# ============================================================================
class AutoEncoder(nn.Module):
    def __init__(self, n_items, n_cate, in_dims, out_dims, act_func='tanh', reparam=True):
        super(AutoEncoder, self).__init__()
        self.n_items = n_items
        self.n_cate = n_cate
        self.reparam = reparam
        
        # 如果n_cate>1，需要类别映射
        if n_cate > 1:
            self.category_map = torch.randint(0, n_items, (n_items,))
            # 重新组织物品，使相同类别的物品连续存储
            self.register_buffer('category_map_buffer', self.category_map)
        
        # 编码器
        encoder_dims = [n_items] + in_dims
        self.encoder_layers = nn.ModuleList()
        for i in range(len(encoder_dims)-1):
            self.encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
        
        # 变分编码层
        if reparam:
            self.fc_mu = nn.Linear(in_dims[-1], in_dims[-1])
            self.fc_logvar = nn.Linear(in_dims[-1], in_dims[-1])
        
        # 解码器
        decoder_dims = [in_dims[-1]] + out_dims + [n_items]
        self.decoder_layers = nn.ModuleList()
        for i in range(len(decoder_dims)-1):
            self.decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
        
        # 激活函数
        self.act_func = self._get_activation(act_func)
        
        self.init_weights()
    
    def _get_activation(self, act_func):
        if act_func == 'tanh':
            return torch.tanh
        elif act_func == 'relu':
            return F.relu
        elif act_func == 'sigmoid':
            return torch.sigmoid
        else:
            return torch.tanh
    
    def init_weights(self):
        for layer in self.encoder_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.decoder_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        if self.reparam:
            size = self.fc_mu.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            self.fc_mu.weight.data.normal_(0.0, std)
            self.fc_mu.bias.data.normal_(0.0, 0.001)
            
            size = self.fc_logvar.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            self.fc_logvar.weight.data.normal_(0.0, std)
            self.fc_logvar.bias.data.normal_(0.0, 0.001)
    
    def Encode(self, x):
        h = x
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i != len(self.encoder_layers) - 1:
                h = self.act_func(h)
        
        if self.reparam:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            return x, z, kl_loss
        else:
            return x, h, torch.tensor(0.0, device=x.device)
    
    def Decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i != len(self.decoder_layers) - 1:
                h = self.act_func(h)
        return h

def compute_ae_loss(recon_x, x):
    # 使用MSE损失作为重建损失
    return F.mse_loss(recon_x, x, reduction='mean')

# ============================================================================
# 核心模型：LDiffRec_a (L-DiffRec in ReChorus)
# ============================================================================
class LDiffRec_a(BaseModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['steps', 'noise_scale', 'reweight', 'n_cate', 'lamda']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--dims', type=str, default='[300]', help='The dims for the MLP.')
        parser.add_argument('--norm', type=int, default=0, help='Normalize the input or not.')
        parser.add_argument('--emb_size', type=int, default=10, help='Timestep embedding size.')
        parser.add_argument('--steps', type=int, default=5, help='Diffusion steps.')
        parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale.')
        parser.add_argument('--noise_min', type=float, default=0.0001, help='Noise lower bound.')
        parser.add_argument('--noise_max', type=float, default=0.02, help='Noise upper bound.')
        parser.add_argument('--sampling_steps', type=int, default=5, help='Steps during inference.')
        parser.add_argument('--reweight', type=int, default=1, help='Assign different weight to different timestep.')
        
        # AutoEncoder 参数
        parser.add_argument('--n_cate', type=int, default=3, help='Number of item categories.')
        parser.add_argument('--in_dims', type=str, default='[300]', help='Encoder dimensions.')
        parser.add_argument('--out_dims', type=str, default='[]', help='Decoder dimensions.')
        parser.add_argument('--act_func', type=str, default='tanh', help='Activation function for AE.')
        parser.add_argument('--lamda', type=float, default=0.03, help='Weight for diffusion loss.')
        parser.add_argument('--reparam', type=int, default=1, help='Use variational AE or not.')
        parser.add_argument('--anneal_cap', type=float, default=0.005, help='Annealing cap for lambda.')
        parser.add_argument('--anneal_steps', type=int, default=500, help='Annealing steps for lambda.')
        parser.add_argument('--vae_anneal_cap', type=float, default=0.3, help='Annealing cap for VAE KL.')
        parser.add_argument('--vae_anneal_steps', type=int, default=200, help='Annealing steps for VAE KL.')
        
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.corpus = corpus

        # 【Magic Fix 1】: 清空 residual_clicked_set
        self.corpus.residual_clicked_set = defaultdict(set)

        # 基础参数
        self.mlp_dims = eval(args.dims)
        self.norm = bool(args.norm)
        self.emb_size = args.emb_size
        self.steps = args.steps
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.sampling_steps = args.sampling_steps if args.sampling_steps > 0 else self.steps
        self.reweight = bool(args.reweight)
        self.lamda = args.lamda
        
        # AutoEncoder 参数
        self.n_cate = args.n_cate
        self.in_dims = eval(args.in_dims)
        self.out_dims = eval(args.out_dims)
        self.act_func = args.act_func
        self.reparam = bool(args.reparam)
        self.anneal_cap = args.anneal_cap
        self.anneal_steps = args.anneal_steps
        self.vae_anneal_cap = args.vae_anneal_cap
        self.vae_anneal_steps = args.vae_anneal_steps
        
        self.test_all = 1 
        
        # 构建模型组件
        # 1. AutoEncoder
        self.autoencoder = AutoEncoder(
            corpus.n_items, 
            self.n_cate,
            self.in_dims,
            self.out_dims,
            self.act_func,
            self.reparam
        ).to(self.device)
        
        # 2. 扩散模型的MLP (在隐空间上操作)
        latent_size = self.in_dims[-1] if len(self.in_dims) > 0 else corpus.n_items
        mlp_out_dims = self.mlp_dims + [latent_size]
        mlp_in_dims = mlp_out_dims[::-1]
        self.dnn = DNN(mlp_in_dims, mlp_out_dims, self.emb_size, self.norm)
        
        # 3. 扩散参数
        self._build_diffusion_params()
        
        # 训练状态跟踪
        self.history_num_per_term = 10
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))
        
        # 更新计数器
        self.update_count = 0
        self.update_count_vae = 0

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

    def forward(self, feed_dict):
        # 原始交互向量
        x_start = feed_dict['vector'].float()
        batch_size = x_start.shape[0]
        
        # 1. AutoEncoder编码
        batch_cate, batch_latent, vae_kl = self.autoencoder.Encode(x_start)
        
        # 2. 扩散过程采样时间步
        ts, pt = self._sample_timesteps(batch_size)
        
        # 3. 向隐变量添加噪声
        noise = torch.randn_like(batch_latent)
        x_t = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, ts, batch_latent.shape) * batch_latent +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, ts, batch_latent.shape) * noise
        )
        
        # 4. 扩散模型预测原始隐变量
        predicted_latent = self.dnn(x_t, ts)
        
        # 5. 计算扩散损失
        mse = mean_flat((batch_latent - predicted_latent) ** 2)
        
        if self.reweight:
            snr = self.alphas_cumprod[ts] / (1 - self.alphas_cumprod[ts])
            snr_prev = self.alphas_cumprod_prev[ts] / (1 - self.alphas_cumprod_prev[ts])
            weight = torch.where((ts == 0), torch.tensor(1.0, device=self.device), snr_prev - snr)
            weight = torch.clamp(weight, min=0)
        else:
            weight = torch.ones_like(mse)
            
        diffusion_losses = weight * mse
        
        # 6. 解码重建
        batch_recon = self.autoencoder.Decode(predicted_latent)
        
        # 7. 计算AutoEncoder损失
        # Lambda退火
        if self.anneal_steps > 0:
            lamda = max((1. - self.update_count / self.anneal_steps) * self.lamda, self.anneal_cap)
        else:
            lamda = max(self.lamda, self.anneal_cap)
        
        # VAE KL退火
        if self.vae_anneal_steps > 0:
            anneal = min(self.vae_anneal_cap, 1. * self.update_count_vae / self.vae_anneal_steps)
        else:
            anneal = self.vae_anneal_cap
        
        ae_loss = compute_ae_loss(batch_recon, batch_cate) + anneal * vae_kl
        
        # 8. 总损失 TODO: 问题很大
        if self.reweight:
            loss = lamda * torch.mean(diffusion_losses / pt) + ae_loss
        else:
            loss = torch.mean(diffusion_losses / pt) + lamda * ae_loss
        
        # 9. 更新历史损失
        with torch.no_grad():
            for t, loss_val in zip(ts, diffusion_losses):
                if self.Lt_count[t] == self.history_num_per_term:
                    self.Lt_history[t] = torch.cat([self.Lt_history[t, 1:], loss_val.unsqueeze(0)])
                else:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val
                    self.Lt_count[t] += 1
        
        self.update_count += 1
        self.update_count_vae += 1
        
        return {'loss': loss, 'prediction': torch.zeros(batch_size, 1, device=self.device)}

    def loss(self, out_dict):
        return out_dict['loss']

    # ========================================================================== #
    #                   扩散模型相关函数 (与DiffRec_c相同)                        #
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
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
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
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        # 这里我们使用x0预测模式
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
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def inference(self, feed_dict):
        batch_users = feed_dict['user_id']
        batch_size = len(batch_users)
        
        # 原始交互向量
        x_start = feed_dict['vector'].float()
        
        with torch.no_grad():
            # 1. 编码到隐空间
            _, batch_latent, _ = self.autoencoder.Encode(x_start)
            
            # 2. 在隐空间进行扩散采样
            latent_recon = self.p_sample(
                model=lambda xt, t: self.dnn(xt, t),
                x_start=batch_latent,
                steps=self.sampling_steps,
                sampling_noise=True
            ).to(self.device)
            
            # 3. 解码回物品空间
            prediction = self.autoencoder.Decode(latent_recon)
        
        # 【Magic Fix 2】: Swap Trick (列交换)
        target_items = feed_dict['item_id'].view(-1)
        batch_indices = torch.arange(batch_size).to(self.device)
        
        if target_items.shape[0] > 0 and target_items[0] != 0:  # 只在评估时交换
            # 2. 取出第 0 列的分数 (备份)
            col0_scores = prediction[:, 0].clone()
            
            # 3. 取出目标列的分数
            target_scores = prediction[batch_indices, target_items]
            
            # 4. 交换
            prediction[:, 0] = target_scores
            prediction[batch_indices, target_items] = col0_scores

        return {'prediction': prediction}

    # ============================================================================
    # DNN模型 (与DiffRec_c相同，但输入维度不同)
    # ============================================================================
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
            target_item = 0
            if self.phase != 'train':
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
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, act_func='tanh'):
        super(DNN, self).__init__()
        self.in_dims = in_dims[0] if isinstance(in_dims, list) else in_dims
        self.out_dims = out_dims
        self.emb_size = emb_size
        self.norm = norm
        
        # 时间嵌入
        self.time_emb = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
        )
        
        # 构建网络层
        dims = [self.in_dims + emb_size] + out_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.drop = nn.Dropout(dropout)
        
        # 激活函数
        if act_func == 'tanh':
            self.activation = torch.tanh
        elif act_func == 'relu':
            self.activation = F.relu
        else:
            self.activation = torch.tanh
        
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
        if self.norm: 
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, time_emb], dim=-1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h

    def _timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: 
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding