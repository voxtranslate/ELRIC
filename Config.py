from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ Global Configuration
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_RD_VALUES: List[float] = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.05]

CONFIG: Dict[str, Any] = {
    # Architecture Dimensions
    "latent_channels":   320,
    "hyper_channels":    192,
    "ds_channels":       128,
    "num_slices":          8,
    "slice_widths":  [20, 20, 30, 35, 45, 50, 55, 65],
    "dict_size":         128,
    "dict_dim":          640,
    "dict_qk_dim":        64,
    "csff_layers":         3,
    "downscale_factor":    4,
    "preview_slices":      4,
    
    # Internal Model Hyperparameters
    "leaky_relu_slope":       0.2,
    "rbb_min_mid_ch":         16,
    "rbb_kernel_size":        3,
    "fsab_pool_kernel":       3,
    "fsab_min_ch":            8,
    "fsab_res_scale_init":    0.1,
    "msca_split_ratio":       3,
    "msca_kernel_large":      5,
    "msca_gate_alpha_init":   0.5,
    "hda_coarse_div":         4,
    "hda_min_coarse":         4,
    "hda_topk_ratio":         0.25,
    "hda_tau_clamp":          0.05,
    "hda_agg_w":              0.7,
    "hda_conf_w":             0.3,
    "hda_init_std":           0.02,
    "gwcf_num_heads":         4,
    "gwcf_window_size":       4,
    "gwcf_complexity_div":    4,
    "sem_sigma_floor":        0.10,

    # Loss Weights & Scaling
    "lambda_rd":         0.013,
    "lambda_preview":    0.10,
    "lambda_ds_match":   0.30,
    "lambda_sr":         0.30,
    "lambda_msssim":     0.84,
    "lambda_lpips":      0.05,
    "lambda_perceptual": 0.0,
    "bpp_floor":          0.05,
    "lambda_bpp_floor":   50.0,
    "bpp_ceiling":        3.0,
    "lambda_bpp_ceiling": 20.0,
    "bpp_hard_div":       5.0,
    
    # Training Loop Parameters
    "clip_max_norm_main":  10.0,
    "clip_max_norm_ent":   50.0,
    "data_dir":          "",
    "kodak_dir":         "",
    "train_ratio":       0.70,
    "val_ratio":         0.15,
    "test_ratio":        0.15,
    "patch_size":        256,
    "color_jitter_b":    0.05,
    "color_jitter_c":    0.05,
    "batch_size":          32,
    "num_workers":         4,
    "lr":                1e-4,
    "lr_aux":            1e-3,
    "num_epochs":        3,
    "val_every":           6,
    "lr_scheduler":      "cosine",
    "lr_patience":         8,
    "lr_factor":          0.5,
    "min_lr":             1e-6,
    "checkpoint_dir":    "",
    "results_dir":       "",
    "vis_dir":           "",
    "kodak_vis_dir":     "",
    "save_every":          10,
    "seed":              42,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
}


def set_seed(seed: int = 42):
    """Enforce strict determinism for identical Kaggle/Colab outputs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)