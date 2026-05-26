import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ Shared Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        mid = max(channels // 2, cfg["rbb_min_mid_ch"])
        slope = cfg["leaky_relu_slope"]
        ks = cfg["rbb_kernel_size"]
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(mid, mid, ks, padding=ks//2),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(mid, channels, 1),
        )
    def forward(self, x):
        return x + self.net(x)


class FreqSpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        ks = cfg["fsab_pool_kernel"]
        slope = cfg["leaky_relu_slope"]
        self.freq_blur = nn.AvgPool2d(ks, stride=1, padding=ks//2)
        self.non_att = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
        )
        self.att = nn.Sequential(
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.cross = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(slope, inplace=True),
        )
        mid_c = max(channels // cfg["fsab_min_ch"], cfg["fsab_min_ch"])
        self.spatial_w = nn.Sequential(
            nn.Conv2d(channels, mid_c, 1),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(mid_c, 2, 1),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1, 1), cfg["fsab_res_scale_init"]))
        self.out_conv  = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x_low  = self.freq_blur(x)
        x_high = x - x_low
        f_non  = self.non_att(x_low)
        f_att  = x_high * self.att(x_high)
        cross  = self.cross(torch.cat([f_non, f_att], dim=1))
        f_non  = f_non + cross
        f_att  = f_att + cross
        w      = F.softmax(self.spatial_w(x), dim=1)
        out    = w[:, 0:1] * f_non + w[:, 1:2] * f_att
        return self.out_conv(out) + self.res_scale * x


class MultiScaleChannelAttention(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        div = cfg["msca_split_ratio"]
        c3 = max(channels // div, 1)
        c5 = max(channels // div, 1)
        cd = max(channels - c3 - c5, 1)
        k5 = cfg["msca_kernel_large"]
        self.path_3x3 = nn.Conv2d(channels, c3, 3, padding=1)
        self.path_5x5 = nn.Conv2d(channels, c5, k5, padding=k5//2)
        self.path_dil = nn.Conv2d(channels, cd, 3, padding=2, dilation=2)
        self.ms_merge = nn.Sequential(nn.Conv2d(c3 + c5 + cd, channels, 1), nn.ReLU(inplace=True))
        self.strip_h  = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
        )
        self.strip_v  = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )
        self.strip_gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.att_gate   = nn.Sequential(
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
            nn.Conv2d(channels, channels, 1), nn.Sigmoid(),
        )
        self.res_path   = nn.Sequential(
            ResidualBottleneckBlock(channels, cfg),
            ResidualBottleneckBlock(channels, cfg),
        )
        self.gate_alpha = nn.Parameter(torch.ones(1) * cfg["msca_gate_alpha_init"])
        self.out        = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        f  = self.ms_merge(torch.cat([self.path_3x3(x), self.path_5x5(x), self.path_dil(x)], dim=1))
        f  = f * self.strip_gate(self.strip_h(f) + self.strip_v(f)) + f
        local_std = f.std(dim=1, keepdim=True).detach()
        norm_std  = (local_std - local_std.mean()) / (local_std.std() + 1e-6)
        cx_weight = torch.sigmoid(norm_std * self.gate_alpha)
        att = self.att_gate(f)
        res = self.res_path(f)
        return self.out(att * (0.5 + 0.5 * cx_weight) * res + res) + x


class AdaptiveMultiScaleConv(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        k5 = cfg["msca_kernel_large"]
        self.lin_in  = nn.Conv2d(channels, channels, 1)
        self.dw_3    = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw_3d2  = nn.Conv2d(channels, channels, 3, padding=2, groups=channels, dilation=2)
        self.dw_5    = nn.Conv2d(channels, channels, k5, padding=k5//2, groups=channels)
        self.mix_w   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, 3), nn.Softmax(dim=1),
        )
        half = max(channels // 2, 1)
        self.dw_hi   = nn.Conv2d(half, half, 3, padding=1, groups=half)
        self.dw_lo   = nn.Conv2d(channels - half, channels - half, k5, padding=k5//2, groups=channels - half)
        self.half    = half
        self.lin_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h  = self.lin_in(x)
        w  = self.mix_w(h)
        mx = (w[:, 0:1].view(-1,1,1,1) * self.dw_3(h)
            + w[:, 1:2].view(-1,1,1,1) * self.dw_3d2(h)
            + w[:, 2:3].view(-1,1,1,1) * self.dw_5(h))
        fa = torch.cat([self.dw_hi(mx[:, :self.half]),
                        self.dw_lo(mx[:, self.half:])], dim=1)
        return self.lin_out(mx + fa)


class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        num_layers = cfg["csff_layers"]
        self.num_layers   = num_layers
        self.amsc_layers  = nn.ModuleList([AdaptiveMultiScaleConv(channels, cfg) for _ in range(num_layers)])
        self.cross_scale  = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1) for _ in range(num_layers)])
        self.scale_imp    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, num_layers + 1), nn.Softmax(dim=1),
        )
        self.merge        = nn.Conv2d(channels * (num_layers + 1), channels, 1)
        self.lap_blur     = nn.AvgPool2d(3, stride=1, padding=1)
        self.lap_att_net  = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        scales = [x]
        curr = x
        for amsc in self.amsc_layers:
            curr = amsc(curr); scales.append(curr)
        enhanced = [scales[0]]
        for i in range(1, len(scales)):
            delta = self.cross_scale[i - 1](torch.cat([scales[i-1], scales[i]], dim=1))
            enhanced.append(scales[i] + delta)
        imp    = self.scale_imp(x)
        merged = self.merge(torch.cat([imp[:, k].view(-1,1,1,1) * enhanced[k]
                                       for k in range(len(enhanced))], dim=1))
        lap    = (x - self.lap_blur(x)).abs().mean(dim=1, keepdim=True)
        att    = self.lap_att_net(lap)
        return att * merged + (1.0 - att) * merged * 0.5


class HierarchicalDictionaryAttention(nn.Module):
    def __init__(self, in_channels: int, cfg: Dict[str, Any]):
        super().__init__()
        dict_size = cfg["dict_size"]
        dict_dim = cfg["dict_dim"]
        qk_dim = cfg["dict_qk_dim"]
        
        n_coarse      = max(dict_size // cfg["hda_coarse_div"], cfg["hda_min_coarse"])
        n_fine        = dict_size - n_coarse
        self.D_c      = nn.Parameter(torch.randn(n_coarse, dict_dim) * cfg["hda_init_std"])
        self.D_f      = nn.Parameter(torch.randn(n_fine,   dict_dim) * cfg["hda_init_std"])
        self.n_coarse = n_coarse
        self.n_fine   = n_fine
        self.topk_ratio = cfg["hda_topk_ratio"]
        self.tau_clamp  = cfg["hda_tau_clamp"]
        self.agg_w      = cfg["hda_agg_w"]
        self.conf_w     = cfg["hda_conf_w"]
        
        self.W_Qc  = nn.Linear(in_channels, qk_dim)
        self.W_Qf  = nn.Linear(in_channels, qk_dim)
        self.W_Kc  = nn.Linear(dict_dim, qk_dim)
        self.W_Kf  = nn.Linear(dict_dim, qk_dim)
        self.tau_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // 8, 4)), nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 8, 4), 1), nn.Softplus(),
        )
        self.conf_c  = nn.Linear(dict_dim, 1)
        self.conf_f  = nn.Linear(dict_dim, 1)
        self.ffn     = nn.Sequential(
            nn.Linear(dict_dim * 2, dict_dim), nn.GELU(),
            nn.Linear(dict_dim, in_channels),
        )
        self.norm     = nn.LayerNorm(in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def _sparse_attn(self, Q, K, V, tau, conf_net, topk):
        sim = (Q @ K.T) / tau.clamp(min=self.tau_clamp)
        if topk < sim.shape[-1]:
            kval, kidx = torch.topk(sim, topk, dim=-1)
            sparse = torch.full_like(sim, float("-inf"))
            sparse.scatter_(-1, kidx, kval)
        else:
            sparse = sim
        A        = F.softmax(sparse, dim=-1)
        agg      = A @ V
        conf     = torch.sigmoid(conf_net(V))
        agg_conf = A @ (V * conf)
        return self.agg_w * agg + self.conf_w * agg_conf

    def forward(self, x_ms):
        B, C, H, W = x_ms.shape
        x_flat = x_ms.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C)
        tau    = self.tau_net(x_ms).unsqueeze(1).expand(B, H*W, 1).reshape(B*H*W, 1)
        Fc = self._sparse_attn(self.W_Qc(x_flat), self.W_Kc(self.D_c),
                               self.D_c, tau, self.conf_c,
                               max(1, int(self.n_coarse * self.topk_ratio)))
        Ff = self._sparse_attn(self.W_Qf(x_flat), self.W_Kf(self.D_f),
                               self.D_f, tau, self.conf_f,
                               max(1, int(self.n_fine * self.topk_ratio)))
        F_out = self.norm(self.ffn(torch.cat([Fc, Ff], dim=-1)) + x_flat)
        return self.out_proj(F_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())


class GatedWindowContextFusion(nn.Module):
    def __init__(self, channels: int, cfg: Dict[str, Any]):
        super().__init__()
        self.channels    = channels
        self.window_size = cfg["gwcf_window_size"]
        self.num_heads   = cfg["gwcf_num_heads"]
        self.scale       = max(channels // self.num_heads, 4) ** -0.5
        gi = channels * 2
        
        self.gate_i  = nn.Sequential(nn.Conv2d(gi, channels, 1), nn.Sigmoid())
        self.gate_f  = nn.Sequential(nn.Conv2d(gi, channels, 1), nn.Sigmoid())
        self.gate_o  = nn.Sequential(nn.Conv2d(gi, channels, 1), nn.Sigmoid())
        self.cell_w  = nn.Sequential(nn.Conv2d(gi, channels, 1), nn.Tanh())
        
        mid_cx = max(channels // cfg["gwcf_complexity_div"], 4)
        self.complexity = nn.Sequential(
            nn.Conv2d(channels, mid_cx, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_cx, 1, 1), nn.Sigmoid(),
        )
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.o_proj = nn.Conv2d(channels, channels, 1)

    def _window_attn(self, q, k, v):
        B, C, H, Ws = q.shape
        ws = min(self.window_size, H, Ws)
        nh, hd = self.num_heads, max(C // self.num_heads, 1)
        pH = (ws - H  % ws) % ws
        pW = (ws - Ws % ws) % ws
        if pH or pW:
            q = F.pad(q, (0, pW, 0, pH))
            k = F.pad(k, (0, pW, 0, pH))
            v = F.pad(v, (0, pW, 0, pH))
        Hp, Wp = q.shape[2], q.shape[3]
        nHw, nWw = Hp // ws, Wp // ws
        
        def part(t):
            t = t.reshape(B, nh, hd, Hp, Wp).permute(0,1,3,4,2).contiguous()
            t = t.reshape(B, nh, nHw, ws, nWw, ws, hd).permute(0,2,4,1,3,5,6).contiguous()
            return t.reshape(B * nHw * nWw, nh, ws * ws, hd)
        
        Qw, Kw, Vw = part(q), part(k), part(v)
        attn = F.softmax((Qw @ Kw.transpose(-2, -1)) * self.scale, dim=-1)
        out  = (attn @ Vw).reshape(B, nHw, nWw, nh, ws, ws, hd)
        
        # Safely reverse the permutation to recover spatial structure
        out = out.permute(0, 3, 6, 1, 4, 2, 5).contiguous().reshape(B, C, Hp, Wp)
        return out[:, :, :H, :Ws]

    def forward(self, context, entropy_state):
        gi   = torch.cat([context, entropy_state], dim=1)
        cell = self.gate_f(gi) * context + self.gate_i(gi) * self.cell_w(gi)
        Q    = self.q_proj(entropy_state * self.complexity(entropy_state))
        attn = self.o_proj(self._window_attn(Q, self.k_proj(cell), self.v_proj(cell)))
        return self.gate_o(gi) * (cell + attn)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ Slice Entropy Model (SEM)
# ─────────────────────────────────────────────────────────────────────────────

class SliceEntropyModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        latent_channels = cfg["latent_channels"]
        hyper_channels  = cfg["hyper_channels"]
        num_slices      = cfg["num_slices"]
        slice_widths    = cfg["slice_widths"]
        
        assert sum(slice_widths) == latent_channels
        assert len(slice_widths) == num_slices
        self.num_slices   = num_slices
        self.slice_widths = slice_widths
        self.sigma_floor  = cfg["sem_sigma_floor"]

        self.ch_context_nets       = nn.ModuleList()
        self.csff_nets             = nn.ModuleList()
        self.hda_nets              = nn.ModuleList()
        self.gwcf_modules          = nn.ModuleList()
        self.param_nets            = nn.ModuleList()
        self.complexity_sigma_nets = nn.ModuleList()

        cumulative_ctx = 0
        slope = cfg["leaky_relu_slope"]
        for i in range(num_slices):
            sc = slice_widths[i]
            self.ch_context_nets.append(
                nn.Identity() if i == 0 else MultiScaleChannelAttention(cumulative_ctx, cfg)
            )
            self.csff_nets.append(CrossScaleFeatureFusion(hyper_channels, cfg))
            self.hda_nets.append(HierarchicalDictionaryAttention(hyper_channels, cfg))
            self.gwcf_modules.append(GatedWindowContextFusion(hyper_channels, cfg))
            
            param_in = hyper_channels * 2 + (cumulative_ctx if i > 0 else 0)
            self.param_nets.append(nn.Sequential(
                nn.Conv2d(param_in, hyper_channels, 1), nn.LeakyReLU(slope, inplace=True),
                nn.Conv2d(hyper_channels, hyper_channels // 2, 1), nn.LeakyReLU(slope, inplace=True),
                nn.Conv2d(hyper_channels // 2, sc * 2, 1),
            ))
            self.complexity_sigma_nets.append(nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, sc, 1), nn.Softplus(),
            ))
            cumulative_ctx += sc

        self.ctx_init = nn.Conv2d(hyper_channels, hyper_channels, 1)
        mid_c = max(hyper_channels // cfg["gwcf_complexity_div"], 4)
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(hyper_channels, mid_c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, 1, 1), nn.Sigmoid(),
        )

    def forward(self, y_slices, hyper_feat, gaussian_conditional, training=True):
        ctx = self.ctx_init(hyper_feat)
        complexity_map = self.complexity_estimator(hyper_feat)
        y_hat_slices, all_likelihoods, decoded = [], [], []
        for i in range(self.num_slices):
            sc = self.slice_widths[i]
            ch_ctx = None if i == 0 else self.ch_context_nets[i](torch.cat(decoded, dim=1))
            
            f_csff = self.csff_nets[i](hyper_feat)
            f_hda  = self.hda_nets[i](f_csff)
            
            param_input = torch.cat([hyper_feat, f_hda, ch_ctx], dim=1) if ch_ctx is not None else torch.cat([hyper_feat, f_hda], dim=1)
            params  = self.param_nets[i](param_input)
            mu_i, sigma_raw = params.chunk(2, dim=1)
            sigma_i  = F.softplus(sigma_raw) + self.sigma_floor
            sigma_scale = self.complexity_sigma_nets[i](complexity_map)
            sigma_i     = sigma_i * (1.0 + sigma_scale)
            
            y_hat_i, likes_i = gaussian_conditional(y_slices[i], sigma_i, means=mu_i)
            y_hat_slices.append(y_hat_i)
            all_likelihoods.append(likes_i)
            decoded.append(y_hat_i.detach())
            
            psi = F.interpolate(params[:, :sc], size=hyper_feat.shape[-2:])
            psi = F.pad(psi, (0, 0, 0, 0, 0, hyper_feat.shape[1] - psi.shape[1]))
            ctx = self.gwcf_modules[i](ctx, psi)
            
        return y_hat_slices, all_likelihoods


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ Encoder / Decoder Networks
# ─────────────────────────────────────────────────────────────────────────────

class DualPathEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], in_ch=3):
        super().__init__()
        M, N, ds_ch = cfg["latent_channels"], cfg["hyper_channels"], cfg["ds_channels"]
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, N, 5, stride=2, padding=2), GDN(N),
            ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
            nn.Conv2d(N, N, 5, stride=2, padding=2), GDN(N),
            FreqSpatialAttentionBlock(N, cfg),
        )
        self.downscale_branch = nn.Sequential(
            nn.Conv2d(N, ds_ch, 1), ResidualBottleneckBlock(ds_ch, cfg),
        )
        self.compress_branch = nn.Sequential(
            nn.Conv2d(N, N, 5, stride=2, padding=2), GDN(N),
            ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
            nn.Conv2d(N, M, 5, stride=2, padding=2), GDN(M),
            FreqSpatialAttentionBlock(M, cfg),
        )
        self.cross_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ds_ch, N), nn.Sigmoid(),
        )

    def forward(self, x):
        sf     = self.shared(x)
        y_dn   = self.downscale_branch(sf)
        gate   = self.cross_att(y_dn).view(sf.shape[0], -1, 1, 1)
        y_comp = self.compress_branch(sf * gate)
        return y_comp, y_dn


class HyperEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        M, N = cfg["latent_channels"], cfg["hyper_channels"]
        self.net = nn.Sequential(
            nn.Conv2d(M, N, 3, stride=1, padding=1), GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2), GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
        )
    def forward(self, y): return self.net(torch.abs(y))


class HyperDecoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        M, N = cfg["latent_channels"], cfg["hyper_channels"]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1), GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            GDN(N * 3 // 2, inverse=True),
            nn.Conv2d(N * 3 // 2, N, 3, padding=1),
        )
    def forward(self, z_hat): return self.net(z_hat)


class CompressionDecoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], out_ch=3):
        super().__init__()
        M, N = cfg["latent_channels"], cfg["hyper_channels"]
        self.net = nn.Sequential(
            FreqSpatialAttentionBlock(M, cfg),
            ResidualBottleneckBlock(M, cfg), ResidualBottleneckBlock(M, cfg), ResidualBottleneckBlock(M, cfg),
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1), GDN(N, inverse=True),
            FreqSpatialAttentionBlock(N, cfg),
            ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1), GDN(N, inverse=True),
            FreqSpatialAttentionBlock(N, cfg),
            ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
            nn.ConvTranspose2d(N, N // 2, 5, stride=2, padding=2, output_padding=1), GDN(N // 2, inverse=True),
            ResidualBottleneckBlock(N // 2, cfg), ResidualBottleneckBlock(N // 2, cfg),
            nn.ConvTranspose2d(N // 2, out_ch, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, y_hat): return self.net(y_hat)


class ProgressivePreviewDecoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], out_ch=3):
        super().__init__()
        total_ch = cfg["latent_channels"]
        N = cfg["hyper_channels"]
        ds_ch = cfg["ds_channels"]
        
        self.preview_slices = cfg["preview_slices"]
        self.slice_widths   = cfg["slice_widths"]
        self.total_slices   = len(self.slice_widths)
        
        self.content_scorer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(total_ch, self.total_slices),
        )
        self.freq_scorer = nn.Sequential(
            nn.Conv2d(total_ch, self.total_slices, 1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.score_blend = nn.Parameter(torch.tensor(0.6))
        self._pi = sum(sorted(self.slice_widths)[-self.preview_slices:])
        self.net = nn.Sequential(
            FreqSpatialAttentionBlock(self._pi, cfg), ResidualBottleneckBlock(self._pi, cfg),
            nn.ConvTranspose2d(self._pi, N, 5, stride=2, padding=2, output_padding=1),
            ResidualBottleneckBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            ResidualBottleneckBlock(N, cfg),
            nn.ConvTranspose2d(N, N // 2, 5, stride=2, padding=2, output_padding=1),
            ResidualBottleneckBlock(N // 2, cfg),
            nn.ConvTranspose2d(N // 2, out_ch, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
        self.ct_proj  = nn.Conv2d(ds_ch, out_ch, 1)
        self.ct_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, y_hat, y_down_dec=None):
        B, M, Hl, Wl = y_hat.shape
        alpha  = torch.sigmoid(self.score_blend)
        scores = alpha * self.content_scorer(y_hat) + (1 - alpha) * self.freq_scorer(y_hat)
        _, topk_idx = torch.topk(scores, self.preview_slices, dim=1)
        splits   = torch.split(y_hat, self.slice_widths, dim=1)
        selected = []
        for b in range(B):
            idx_b  = topk_idx[b].tolist()
            chosen = [splits[j][b:b+1] for j in idx_b]
            sel    = torch.cat(chosen, dim=1)
            if sel.shape[1] < self._pi:
                sel = F.pad(sel, (0, 0, 0, 0, 0, self._pi - sel.shape[1]))
            else:
                sel = sel[:, :self._pi]
            selected.append(sel)
        sel_batch = torch.cat(selected, dim=0)
        out = self.net(sel_batch)
        if y_down_dec is not None:
            hint = torch.sigmoid(self.ct_proj(
                F.interpolate(y_down_dec, size=out.shape[-2:], mode="bilinear", align_corners=False)))
            out = (out + self.ct_scale * hint).clamp(0, 1)
        return out

    def decode_k_slices(self, y_hat, k_slices: int, y_down_dec=None):
        B, M, Hl, Wl = y_hat.shape
        k = max(1, min(k_slices, self.total_slices))
        alpha  = torch.sigmoid(self.score_blend)
        scores = alpha * self.content_scorer(y_hat) + (1 - alpha) * self.freq_scorer(y_hat)
        _, topk_idx = torch.topk(scores, k, dim=1)
        splits   = torch.split(y_hat, self.slice_widths, dim=1)
        selected = []
        for b in range(B):
            idx_b  = topk_idx[b].tolist()
            chosen = [splits[j][b:b+1] for j in idx_b]
            sel    = torch.cat(chosen, dim=1)
            if sel.shape[1] < self._pi:
                sel = F.pad(sel, (0, 0, 0, 0, 0, self._pi - sel.shape[1]))
            else:
                sel = sel[:, :self._pi]
            selected.append(sel)
        sel_batch = torch.cat(selected, dim=0)
        out = self.net(sel_batch)
        if y_down_dec is not None:
            hint = torch.sigmoid(self.ct_proj(
                F.interpolate(y_down_dec, size=out.shape[-2:], mode="bilinear", align_corners=False)))
            out = (out + self.ct_scale * hint).clamp(0, 1)
        return out


class DownscaleEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], out_ch=3):
        super().__init__()
        ds_ch = cfg["ds_channels"]
        slope = cfg["leaky_relu_slope"]
        self.net = nn.Sequential(
            FreqSpatialAttentionBlock(ds_ch, cfg),
            ResidualBottleneckBlock(ds_ch, cfg),
            ResidualBottleneckBlock(ds_ch, cfg),
            ResidualBottleneckBlock(ds_ch, cfg),
            nn.Conv2d(ds_ch, ds_ch // 2, 3, padding=1), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ds_ch // 2, ds_ch // 4, 3, padding=1), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ds_ch // 4, out_ch, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, y): return self.net(y)


class SuperResolutionDecoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], out_ch=3):
        super().__init__()
        ds_ch = cfg["ds_channels"]
        N = cfg["hyper_channels"]
        scale_factor = cfg["downscale_factor"]
        slope = cfg["leaky_relu_slope"]
        
        num_up = int(math.log2(scale_factor))
        layers: List[nn.Module] = [
            nn.Conv2d(3, N, 3, padding=1),
            FreqSpatialAttentionBlock(N, cfg), ResidualBottleneckBlock(N, cfg),
        ]
        for _ in range(num_up):
            layers += [
                nn.ConvTranspose2d(N, N, 4, stride=2, padding=1),
                nn.LeakyReLU(slope, inplace=True),
                ResidualBottleneckBlock(N, cfg), FreqSpatialAttentionBlock(N, cfg),
            ]
        layers += [nn.Conv2d(N, out_ch, 3, padding=1)]
        self.net          = nn.Sequential(*layers)
        self.scale_factor = scale_factor

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return (base + self.net(x)).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ ELRIC v2 Main Model
# ─────────────────────────────────────────────────────────────────────────────

class ELRIC(nn.Module):
    """ELRIC v2 – heavily config-driven without local magic numbers."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        M, N = cfg["latent_channels"], cfg["hyper_channels"]
        ds_ch = cfg["ds_channels"]
        self.num_slices = cfg["num_slices"]
        self.slice_widths = cfg["slice_widths"]
        slope = cfg["leaky_relu_slope"]

        self.dpe = DualPathEncoder(cfg)
        self.he  = HyperEncoder(cfg)
        self.hd  = HyperDecoder(cfg)
        self.entropy_bottleneck   = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.sem = SliceEntropyModel(cfg)
        self.latent_to_down_feat = nn.Sequential(
            nn.ConvTranspose2d(M, ds_ch, 4, stride=2, padding=1), nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ds_ch, ds_ch, 4, stride=2, padding=1), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ds_ch, ds_ch, 3, padding=1),
        )
        self.cd  = CompressionDecoder(cfg)
        self.ppd = ProgressivePreviewDecoder(cfg)
        self.de  = DownscaleEncoder(cfg)
        self.srd = SuperResolutionDecoder(cfg)

    def forward(self, x):
        y_compress, _  = self.dpe(x)
        z              = self.he(y_compress)
        z_hat, z_likes = self.entropy_bottleneck(z)
        hyper_feat     = self.hd(z_hat)
        y_slices = list(torch.split(y_compress, self.slice_widths, dim=1))
        
        y_hat_sl, y_likes = self.sem(
            y_slices, hyper_feat, self.gaussian_conditional, training=self.training)
        
        y_hat       = torch.cat(y_hat_sl, dim=1)
        y_down_dec  = self.latent_to_down_feat(y_hat)
        x_downscaled = self.de(y_down_dec)
        return {
            "x_hat_compress": self.cd(y_hat),
            "x_hat_sr":       self.srd(x_downscaled),
            "x_preview":      self.ppd(y_hat, y_down_dec),
            "x_downscaled":   x_downscaled,
            "likelihoods_y":  y_likes,
            "likelihoods_z":  z_likes,
            "_y_compress":    y_compress,
            "_y_down_dec":    y_down_dec,
            "_hyper_feat":    hyper_feat,
        }

    @torch.no_grad()
    def compress(self, x):
        y_compress, _ = self.dpe(x)
        z = self.he(y_compress)
        z_hat, z_strings = self.entropy_bottleneck.compress(z)
        hyper_feat = self.hd(z_hat)
        y_hat_sl, _ = self.sem(
            list(torch.split(y_compress, self.slice_widths, dim=1)),
            hyper_feat, self.gaussian_conditional, training=False)
        y_hat = torch.cat(y_hat_sl, dim=1)
        return {"z_strings": z_strings, "y_hat": y_hat, "shape_z": z.shape[-2:]}

    @torch.no_grad()
    def decompress(self, compressed):
        hyper_feat = self.hd(self.entropy_bottleneck.decompress(
            compressed["z_strings"], compressed["shape_z"]))
        y_hat      = compressed["y_hat"]
        y_down_dec = self.latent_to_down_feat(y_hat)
        x_ds       = self.de(y_down_dec)
        return {
            "x_hat_compress": self.cd(y_hat),
            "x_hat_sr":       self.srd(x_ds),
            "x_preview":      self.ppd(y_hat, y_down_dec),
            "x_downscaled":   x_ds,
        }

    def get_parameter_groups(self):
        main = [p for n, p in self.named_parameters() if not n.endswith(".quantiles")]
        aux  = [p for n, p in self.named_parameters() if     n.endswith(".quantiles")]
        return main, aux
