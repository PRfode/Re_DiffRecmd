# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.BaseModel import BaseModel
from collections import defaultdict

# ============================================================================
# Math Utils (Directly from Original Paper's gaussian_diffusion.py)
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
# T-DiffRec Implementation
# ============================================================================
class TDiffRec_b(BaseModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['steps', 'noise_scale', 'reweight', 'w_min', 'w_max']

    @staticmethod
    def parse_model_args(parser):
        # Neural Network Params
        parser.add_argument('--dims', type=str, default='[1000]', help='The dims for the DNN.')
        parser.add_argument('--norm', type=int, default=0, help='Normalize the input or not.')
        parser.add_argument('--emb_size', type=int, default=10, help='Timestep embedding size.')
        
        # Diffusion Params
        parser.add_argument('--steps', type=int, default=100, help='Total diffusion steps (T).')
        parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale.')
        parser.add_argument('--noise_min', type=float, default=0.0001, help='Noise lower bound.')
        parser.add_argument('--noise_max', type=float, default=0.02, help='Noise upper bound.')
        parser.add_argument('--sampling_steps', type=int, default=0, help='Inference steps (T\').')
        parser.add_argument('--reweight', type=int, default=1, help='Assign different weight to different timestep.')
        
        # T-DiffRec Specific Params (Weight Schedule)
        parser.add_argument('--w_min', type=float, default=0.1, help='Minimum weight for earliest interactions.')
        parser.add_argument('--w_max', type=float, default=1.0, help='Maximum weight for latest interactions.')
        
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.corpus = corpus

        # [Framework Adaption]: Clear residual_clicked_set
        # ReChorus by default masks validation/test items. We need to unmask them 
        # because our leave-one-out strategy predicts exactly these items.
        self.corpus.residual_clicked_set = defaultdict(set)

        self.dims = eval(args.dims)
        self.norm = bool(args.norm)
        self.emb_size = args.emb_size
        self.steps = args.steps
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.sampling_steps = args.sampling_steps if args.sampling_steps > 0 else self.steps
        self.reweight = bool(args.reweight)
        
        self.w_min = args.w_min
        self.w_max = args.w_max
        
        self.test_all = 1 

        self.dnn = DNN(corpus.n_items, self.dims, self.emb_size, self.norm)
        self._build_diffusion_params()
        
        self.history_num_per_term = 10
        # Register buffers for Importance Sampling
        self.register_buffer('Lt_history', torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64))
        self.register_buffer('Lt_count', torch.zeros(self.steps, dtype=torch.long))

    def _build_diffusion_params(self):
        """Constructs Beta, Alpha, and Posterior parameters."""
        if self.noise_scale == 0:
            self.betas = torch.tensor([0.0] * self.steps).float().to(self.device)
        else:
            # Linear Variance Schedule (Matched with original gaussian_diffusion.py)
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            map_variance = np.linspace(start, end, self.steps, dtype=np.float64)
            betas = betas_from_linear_variance(self.steps, map_variance)
            betas[0] = 0.00001 # Fixed first beta
            self.register_buffer('betas', torch.tensor(betas).float())
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), alphas_cumprod[:-1]])
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Posterior Variance (Used for denoising)
        # Note: Original code clips log variance. We do the same.
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped', torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        ))
        
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _sample_timesteps(self, batch_size):
        # Importance Sampling Logic (Matched with original)
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

    # ========================================================================
    # Training
    # ========================================================================
    def forward(self, feed_dict):
        x_start = feed_dict['vector'].float()
        batch_size = x_start.shape[0]
        
        # 1. Sample t
        ts, pt = self._sample_timesteps(batch_size)
        noise = torch.randn_like(x_start)
        
        # 2. Add Noise (q_sample)
        x_t = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, ts, x_start.shape) * x_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, ts, x_start.shape) * noise
        )
        
        # 3. Denoise (p_loss)
        predicted_x_start = self.dnn(x_t, ts)
        
        # 4. Calculate Loss
        mse = mean_flat((x_start - predicted_x_start) ** 2)
        
        if self.reweight:
            snr = self.alphas_cumprod[ts] / (1 - self.alphas_cumprod[ts])
            snr_prev = self.alphas_cumprod_prev[ts] / (1 - self.alphas_cumprod_prev[ts])
            weight = torch.where((ts == 0), torch.tensor(1.0, device=self.device), snr_prev - snr)
            weight = torch.clamp(weight, min=0)
        else:
            weight = torch.ones_like(mse)
            
        losses = weight * mse
        
        # 5. Update history for importance sampling
        with torch.no_grad():
            for t, loss_val in zip(ts, losses):
                if self.Lt_count[t] == self.history_num_per_term:
                    self.Lt_history[t] = torch.cat([self.Lt_history[t, 1:], loss_val.unsqueeze(0)])
                else:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val
                    self.Lt_count[t] += 1

        final_loss = torch.mean(losses / pt)
        
        # ReChorus compatibility: dummy prediction
        return {'loss': final_loss, 'prediction': torch.zeros(batch_size, 1, device=self.device)}

    def loss(self, out_dict):
        return out_dict['loss']

    # ========================================================================
    # Inference (Strict alignment with p_sample logic)
    # ========================================================================
    def q_sample_for_inference(self, x_start, t):
        # Add noise to input history, similar to original p_sample when steps < total
        noise = torch.randn_like(x_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def inference(self, feed_dict):
        # We start from user history, corrupt it to step T', and then denoise
        x_start = feed_dict['vector'].float()
        batch_size = x_start.shape[0]
        
        # 1. Forward Corruption (If T' > 0)
        if self.sampling_steps > 0:
            steps_tensor = torch.tensor([self.sampling_steps - 1] * batch_size).to(self.device)
            x = self.q_sample_for_inference(x_start, steps_tensor)
            indices = list(range(self.sampling_steps))[::-1]
        else:
            x = x_start # Zero shot (usually not used)
            indices = []

        # 2. Reverse Denoising (Deterministic p_sample)
        with torch.no_grad():
            for i in indices:
                t = torch.tensor([i] * batch_size).to(self.device)
                
                # Predict x0
                pred_x0 = self.dnn(x, t)
                # Clip prediction to [-1, 1] for stability (Common trick in diffusion)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                # Compute Posterior Mean (No Noise added here for Deterministic Inf)
                x = (
                    self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_x0 +
                    self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
                )
        
        # [Optimization]: Use pred_x0 of the last step as it's cleaner than mean
        final_scores = pred_x0 
        
        # [Framework Adaption]: Swap Trick to fit ReChorus evaluate format
        # Move target item score to column 0
        target_items = feed_dict['item_id'].view(-1)
        col0_scores = final_scores[:, 0].clone()
        batch_indices = torch.arange(batch_size).to(self.device)
        target_scores = final_scores[batch_indices, target_items]
        
        final_scores[:, 0] = target_scores
        final_scores[batch_indices, target_items] = col0_scores

        return {'prediction': final_scores}

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    # ========================================================================
    # Dataset Class
    # ========================================================================
    class Dataset(BaseModel.Dataset):
        def __init__(self, model, corpus, phase):
            self.model = model
            self.corpus = corpus
            self.phase = phase
            self.buffer_dict = dict()
            
            # Precompute history for all phases
            train_df = corpus.data_df['train']
            self.user_history_lists = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
            
            if phase == 'train':
                users = list(self.user_history_lists.keys())
            else:
                users = corpus.data_df[phase]['user_id'].unique().tolist()
                
            self.data = {'user_id': users}

        def __len__(self):
            return len(self.data['user_id'])

        def _get_feed_dict(self, index):
            user_id = self.data['user_id'][index]
            
            # --- T-DiffRec Weighting Logic ---
            items = self.user_history_lists.get(user_id, [])
            vector = np.zeros(self.corpus.n_items, dtype=np.float32)
            
            if len(items) > 0:
                # Linearly scale weights from w_min to w_max
                w_min = self.model.w_min
                w_max = self.model.w_max
                weights = np.linspace(w_min, w_max, num=len(items))
                vector[items] = weights
            
            # Find target for Dev/Test (For swap trick)
            target_item = 0
            if self.phase != 'train':
                user_rows = self.corpus.data_df[self.phase]
                target_df = user_rows[user_rows['user_id'] == user_id]
                if len(target_df) > 0:
                    target_item = target_df['item_id'].values[0]

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

# ============================================================================
# DNN Class (Strict Copy from original DNN.py + init_weights)
# ============================================================================
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
        
        # Init Weights (Important!)
        self.init_weights()
        
    def init_weights(self):
        for layer in self.layers:
            # Xavier Normal
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            # Bias Normal
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