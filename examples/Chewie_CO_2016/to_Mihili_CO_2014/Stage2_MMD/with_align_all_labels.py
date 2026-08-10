import os
import sys
import csv
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import yaml
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from einops import rearrange

from TCLA.data.data_multisession import get_session_data_and_configs, SingleSessionDataset
from TCLA.networks.count_wrapper import CountWrapper
from TCLA.networks.blocks import MultiSessionAutoEncoder


# ==============================================================================
# 1. Config
# ==============================================================================

SOURCE_SESSION = os.environ.get("SOURCE_SESSION_ID", "session_0")
TARGET_SESSION = os.environ.get("TARGET_SESSION_ID", "session_0")
TARGET_MODEL_SESSION = os.environ.get("TARGET_MODEL_SESSION_ID", f"Mihili_CO_2014_{TARGET_SESSION}")
PAIR_TAG = f"{SOURCE_SESSION}_to_{TARGET_SESSION}"


def format_tag(value):
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return str(value).replace(".", "p")


cfg_yaml = f"""
model:
  C_shared: 30
  C: 256
  C_latent: 16
  num_blocks: 4
  num_blocks_decoder: 0
  num_lin_per_mlp: 2
  bidirectional: False
  use_pnba_read_adapter: True
  pnba_proj_channels: 32
  pnba_pooled_neurons: 16

dataset:
  source_datapath_root: {REPO_ROOT}/data/Chewie_CO_2016
  target_datapath_root: {REPO_ROOT}/data/Mihili_CO_2014
  signal_length: 210

adapt:
  lr: 0.001
  num_epochs: 300
  num_warmup_epochs: 100
  batch_size: 16
  num_workers: 4
  grad_clip: 1.0
  random_seed: 42
  mask_prob: 0
  reset_target_read_adapter: False

  eta_mmd: 10
  kernel_mul: 2.0
  kernel_num: 5
  max_mmd_samples_per_condition: 4096

exp:
  source_behavior_checkpoint: {REPO_ROOT}/outputs/Chewie_CO_2016/AE_model/source_{SOURCE_SESSION}_direct_shift_experiment/model/full_ae_behavior_model_final.pt
  save_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/to_Mihili_CO_2014/Stage2_MMD/results_use_pnba_read_adapter_without_reset/{PAIR_TAG}/model/
  loss_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/to_Mihili_CO_2014/Stage2_MMD/results_use_pnba_read_adapter_without_reset/{PAIR_TAG}/loss/
  eval_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/to_Mihili_CO_2014/Stage2_MMD/results_use_pnba_read_adapter_without_reset/{PAIR_TAG}/eval/
"""
cfg = OmegaConf.create(yaml.safe_load(cfg_yaml))
cfg.adapt.num_epochs = int(os.environ.get("ADAPT_NUM_EPOCHS", cfg.adapt.num_epochs))
cfg.adapt.num_warmup_epochs = int(os.environ.get("ADAPT_NUM_WARMUP_EPOCHS", cfg.adapt.num_warmup_epochs))
cfg.adapt.batch_size = int(os.environ.get("BATCH_SIZE", cfg.adapt.batch_size))
cfg.adapt.num_workers = int(os.environ.get("NUM_WORKERS", cfg.adapt.num_workers))

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# ==============================================================================
# 2. Model head and utilities
# ==============================================================================

class BehaviorHead(nn.Module):
    def __init__(self, C_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C_latent, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 2, kernel_size=1),
        )

    def forward(self, z):
        return self.net(z)


def load_torch_object(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def prepare_behavior_target(behavior, pred):
    behavior = behavior.float()
    return behavior if behavior.shape[1] == 2 else behavior.permute(0, 2, 1).contiguous()


def compute_behavior_zscore_stats(behavior, train_indices, eps=1e-6):
    train_behavior = behavior[train_indices]
    behavior_mean = train_behavior.mean(axis=(0, 1), keepdims=True)
    behavior_var = train_behavior.var(axis=(0, 1), keepdims=True)
    behavior_std = np.sqrt(np.maximum(behavior_var, eps))
    return behavior_mean, behavior_var, behavior_std


def normalize_behavior(behavior, behavior_mean, behavior_std):
    return ((behavior - behavior_mean) / behavior_std).astype(np.float32)




def remap_target_labels_to_source(labels):
    labels = np.asarray(labels).astype(np.int64)
    return ((1 - labels) % 8).astype(np.int64)

def apply_notebook_mask(signal, mask_prob):
    if mask_prob > 0:
        mask = (torch.rand_like(signal) > mask_prob).float()
        input_signal = signal * (mask / (1 - mask_prob))
        unmasked = 1 - mask
    else:
        input_signal = signal
        unmasked = torch.ones_like(signal)

    return input_signal, unmasked


def mmd_multikernel(source, target, device, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if source is None or target is None or source.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=device)

    n_s = source.size(0)
    n_t = target.size(0)
    n_samples = n_s + n_t
    if n_samples <= 2:
        return torch.tensor(0.0, device=device)

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    l2_distance_sq = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance_sq.detach()) / (n_samples ** 2 - n_samples)

    bandwidth = torch.clamp(bandwidth, min=1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = sum(torch.exp(-l2_distance_sq / torch.clamp(bw, min=1e-6)) for bw in bandwidth_list)

    xx = kernels[:n_s, :n_s].mean()
    yy = kernels[n_s:, n_s:].mean()
    xy = kernels[:n_s, n_s:].mean()
    return xx + yy - 2 * xy


def maybe_subsample_rows(x, max_rows):
    if max_rows is None or x.size(0) <= max_rows:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:max_rows]
    return x[idx]


def get_condition_latents(z, labels, condition_ids, max_samples_per_condition=None):
    out = {}
    labels = labels.to(z.device)
    for condition in condition_ids:
        mask = labels == condition
        if mask.sum() == 0:
            out[int(condition.item())] = None
            continue
        z_condition = z[mask]
        z_condition = rearrange(z_condition, "b c t -> (b t) c")
        z_condition = maybe_subsample_rows(z_condition, max_samples_per_condition)
        out[int(condition.item())] = z_condition
    return out


def conditional_mmd_loss(
    z_source,
    labels_source,
    z_target,
    labels_target,
    device,
    kernel_mul=2.0,
    kernel_num=5,
    max_samples_per_condition=None,
):
    labels_source = labels_source.to(device)
    labels_target = labels_target.to(device)
    condition_ids = torch.unique(torch.cat([labels_source, labels_target], dim=0))

    source_latents = get_condition_latents(
        z_source,
        labels_source,
        condition_ids,
        max_samples_per_condition=max_samples_per_condition,
    )
    target_latents = get_condition_latents(
        z_target,
        labels_target,
        condition_ids,
        max_samples_per_condition=max_samples_per_condition,
    )

    loss = torch.tensor(0.0, device=device)
    valid_conditions = 0
    for condition in condition_ids:
        key = int(condition.item())
        z_s_d = source_latents.get(key)
        z_t_d = target_latents.get(key)
        if z_s_d is None or z_t_d is None:
            continue
        loss = loss + mmd_multikernel(
            z_s_d,
            z_t_d,
            device=device,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
        )
        valid_conditions += 1

    return loss


def split_target_20_20_60(spikes, labels, seed=42):
    indices = np.arange(len(spikes))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.6,
        random_state=seed,
        stratify=labels,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.5,
        random_state=seed,
        stratify=labels[train_val_indices],
    )
    return train_indices, val_indices, test_indices


def split_source_80_10_10(spikes, labels, seed=42):
    indices = np.arange(len(spikes))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=labels[temp_indices],
    )
    return train_indices, val_indices, test_indices


def make_loader(spikes, behavior, labels, indices, session_id, batch_size, num_workers, shuffle):
    dataset = SingleSessionDataset(
        spikes[indices],
        behavior[indices],
        labels[indices],
        session_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def build_shared_ae_config(model_cfg, signal_length):
    return {
        "C_in": model_cfg.C_shared,
        "C": model_cfg.C,
        "C_latent": model_cfg.C_latent,
        "L": signal_length,
        "num_blocks": model_cfg.num_blocks,
        "bidirectional": model_cfg.bidirectional,
        "num_blocks_decoder": model_cfg.num_blocks_decoder,
        "num_lin_per_mlp": model_cfg.num_lin_per_mlp,
        "use_pnba_read_adapter": model_cfg.get("use_pnba_read_adapter", False),
        "pnba_proj_channels": model_cfg.get("pnba_proj_channels", 32),
        "pnba_pooled_neurons": model_cfg.get("pnba_pooled_neurons", 8),
        "latent_smooth_sigma_bins": model_cfg.get("latent_smooth_sigma_bins", 0.0),
    }


def build_countwrapped_ae(session_configs, session_ids, shared_ae_config):
    selected_configs = {sid: session_configs[sid] for sid in session_ids}
    model = MultiSessionAutoEncoder(selected_configs, dict(shared_ae_config))
    return CountWrapper(model)


def load_ae_state(model, state_dict, allowed_missing_prefixes=None):
    allowed_missing_prefixes = allowed_missing_prefixes or []
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]

    if unexpected or bad_missing:
        raise RuntimeError(
            "Failed to load AE checkpoint.\n"
            f"Unexpected keys: {unexpected}\n"
            f"Missing keys: {bad_missing}"
        )
    return result


def freeze_all_but_read_adapters(model, target_session_id):
    for param in model.parameters():
        param.requires_grad = False

    adaptation_params = model.ae_net.get_finetune_parameters(target_session_id)
    for param in adaptation_params:
        param.requires_grad = True

    return adaptation_params


def reset_module_parameters(module):
    for layer in module.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def reset_target_read_adapter(model, target_session_id):
    if model.ae_net.use_pnba_read_adapter:
        reset_module_parameters(model.ae_net.pnba_readin)
        reset_module_parameters(model.ae_net.pnba_readout)
        return [
            "ae_net.pnba_readin",
            "ae_net.pnba_readout",
        ]

    reset_module_parameters(model.ae_net.read_in_layers[target_session_id])
    reset_module_parameters(model.ae_net.read_out_layers[target_session_id])
    return [
        f"ae_net.read_in_layers.{target_session_id}",
        f"ae_net.read_out_layers.{target_session_id}",
    ]


def plot_loss_curve(train_history, val_history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = np.arange(1, len(train_history) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(epochs, [x["total"] for x in train_history], label="Train total")
    ax.plot(epochs, [x["total"] for x in val_history], label="Val total", linestyle="--")
    ax.plot(epochs, [x["recon"] for x in train_history], label="Train recon", alpha=0.75)
    ax.plot(epochs, [x["recon"] for x in val_history], label="Val recon", alpha=0.75, linestyle="--")
    ax.plot(epochs, [x["mmd"] for x in train_history], label="Train MMD", alpha=0.75)
    ax.plot(epochs, [x["mmd"] for x in val_history], label="Val MMD", alpha=0.75, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Target session adaptation with conditional MMD")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def evaluate_behavior_r2(target_ae, behavior_head, data_loader, target_session_id, device):
    target_ae.eval()
    behavior_head.eval()

    true_trials = []
    pred_trials = []
    for batch in tqdm(data_loader, desc="Evaluating target test R2"):
        signal = batch["signal"].to(device)
        behavior = batch["behavior"].to(device)
        _, z = target_ae(signal, target_session_id)
        pred = behavior_head(z)
        true = prepare_behavior_target(behavior, pred)
        true_trials.append(true.cpu())
        pred_trials.append(pred.cpu())

    y_true_trials = torch.cat(true_trials, dim=0).numpy()
    y_pred_trials = torch.cat(pred_trials, dim=0).numpy()
    y_true_flat = np.transpose(y_true_trials, (0, 2, 1)).reshape(-1, 2)
    y_pred_flat = np.transpose(y_pred_trials, (0, 2, 1)).reshape(-1, 2)

    r2_raw = r2_score(y_true_flat, y_pred_flat, multioutput="raw_values")
    r2_uniform = r2_score(y_true_flat, y_pred_flat, multioutput="uniform_average")

    return {
        "r2_x": float(r2_raw[0]),
        "r2_y": float(r2_raw[1]),
        "r2_uniform": float(r2_uniform),
    }, y_true_trials, y_pred_trials


# ==============================================================================
# 3. Main adaptation
# ==============================================================================

def run_target_adaptation_conditional_mmd(cfg, target_session_id, source_session_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.adapt.random_seed)
    np.random.seed(cfg.adapt.random_seed)

    os.makedirs(cfg.exp.save_dir, exist_ok=True)
    os.makedirs(cfg.exp.loss_dir, exist_ok=True)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)

    ckpt = load_torch_object(cfg.exp.source_behavior_checkpoint, map_location="cpu")
    if "ae_state_dict" not in ckpt or "behavior_head_state_dict" not in ckpt:
        raise RuntimeError("Please use full_ae_behavior_model_final.pt with ae_state_dict and behavior_head_state_dict.")

    stage1_cfg = OmegaConf.create(ckpt.get("config", {}))
    if "model" in stage1_cfg:
        cfg.model = stage1_cfg.model
    if "dataset" in stage1_cfg and "datapath_root" in stage1_cfg.dataset:
        cfg.dataset.source_datapath_root = stage1_cfg.dataset.datapath_root
    if "dataset" in stage1_cfg and "signal_length" in stage1_cfg.dataset:
        cfg.dataset.signal_length = stage1_cfg.dataset.signal_length

    target_model_session_id = TARGET_MODEL_SESSION

    source_session_data, source_session_configs = get_session_data_and_configs(
        [source_session_id],
        cfg.dataset.source_datapath_root,
    )
    target_session_data, target_session_configs = get_session_data_and_configs(
        [target_session_id],
        cfg.dataset.target_datapath_root,
    )

    source_data = source_session_data[source_session_id]
    target_data = target_session_data[target_session_id]
    all_session_configs = {
        source_session_id: source_session_configs[source_session_id],
        target_model_session_id: target_session_configs[target_session_id],
    }
    spikes_src = source_data["spikes"]
    behavior_src = source_data["behavior"]
    labels_src = source_data["label"]
    spikes_tgt = target_data["spikes"]
    behavior_tgt = target_data["behavior"]
    labels_tgt_original = target_data["label"].astype(np.int64)
    labels_tgt = remap_target_labels_to_source(labels_tgt_original)

    source_train_idx, source_val_idx, source_test_idx = split_source_80_10_10(
        spikes_src,
        labels_src,
        seed=cfg.adapt.random_seed,
    )
    target_train_idx, target_val_idx, target_test_idx = split_target_20_20_60(
        spikes_tgt,
        labels_tgt,
        seed=cfg.adapt.random_seed,
    )

    source_behavior_mean, source_behavior_var, source_behavior_std = compute_behavior_zscore_stats(
        behavior_src,
        source_train_idx,
    )
    target_behavior_mean, target_behavior_var, target_behavior_std = compute_behavior_zscore_stats(
        behavior_tgt,
        target_train_idx,
    )
    behavior_src = normalize_behavior(
        behavior_src,
        source_behavior_mean,
        source_behavior_std,
    )
    behavior_tgt = normalize_behavior(
        behavior_tgt,
        target_behavior_mean,
        target_behavior_std,
    )
    cfg.dataset.behavior_normalized = True
    cfg.dataset.source_behavior_mean = source_behavior_mean.reshape(-1).tolist()
    cfg.dataset.source_behavior_var = source_behavior_var.reshape(-1).tolist()
    cfg.dataset.source_behavior_std = source_behavior_std.reshape(-1).tolist()
    cfg.dataset.target_behavior_mean = target_behavior_mean.reshape(-1).tolist()
    cfg.dataset.target_behavior_var = target_behavior_var.reshape(-1).tolist()
    cfg.dataset.target_behavior_std = target_behavior_std.reshape(-1).tolist()

    np.savez(
        os.path.join(cfg.exp.eval_dir, "target_split_indices_20_20_60.npz"),
        train_indices=target_train_idx,
        val_indices=target_val_idx,
        test_indices=target_test_idx,
        target_session_id=target_session_id,
        target_model_session_id=target_model_session_id,
        source_session_id=source_session_id,
        behavior_normalized=True,
        target_behavior_mean=target_behavior_mean,
        target_behavior_var=target_behavior_var,
        target_behavior_std=target_behavior_std,
    )
    np.savez(
        os.path.join(cfg.exp.eval_dir, "source_split_indices_80_10_10.npz"),
        train_indices=source_train_idx,
        val_indices=source_val_idx,
        test_indices=source_test_idx,
        source_session_id=source_session_id,
        behavior_normalized=True,
        source_behavior_mean=source_behavior_mean,
        source_behavior_var=source_behavior_var,
        source_behavior_std=source_behavior_std,
    )

    print("===================================================")
    print("Behavior normalization: enabled")
    print(f"  source mean/std from source train split: {cfg.dataset.source_behavior_mean} / {cfg.dataset.source_behavior_std}")
    print(f"  target mean/std from target train split: {cfg.dataset.target_behavior_mean} / {cfg.dataset.target_behavior_std}")
    print("Spike/rate normalization: disabled")
    print(f"Source session: {source_session_id} | total trials: {len(spikes_src)}")
    print(f"Source train trials for conditional MMD: {len(source_train_idx)}")
    print(f"Source val trials for conditional MMD:   {len(source_val_idx)}")
    print(f"Source test trials:                      {len(source_test_idx)}")
    print(f"Target session: {target_session_id} | total trials: {len(spikes_tgt)}")
    print(f"Target train trials: {len(target_train_idx)}")
    print(f"Target val trials:   {len(target_val_idx)}")
    print(f"Target test trials:  {len(target_test_idx)}")
    print("===================================================")

    source_train_loader = make_loader(
        spikes_src,
        behavior_src,
        labels_src,
        source_train_idx,
        source_session_id,
        cfg.adapt.batch_size,
        cfg.adapt.num_workers,
        shuffle=True,
    )
    source_val_loader = make_loader(
        spikes_src,
        behavior_src,
        labels_src,
        source_val_idx,
        source_session_id,
        cfg.adapt.batch_size,
        cfg.adapt.num_workers,
        shuffle=True,
    )
    target_train_loader = make_loader(
        spikes_tgt,
        behavior_tgt,
        labels_tgt,
        target_train_idx,
        target_model_session_id,
        cfg.adapt.batch_size,
        cfg.adapt.num_workers,
        shuffle=True,
    )
    target_val_loader = make_loader(
        spikes_tgt,
        behavior_tgt,
        labels_tgt,
        target_val_idx,
        target_model_session_id,
        cfg.adapt.batch_size,
        cfg.adapt.num_workers,
        shuffle=False,
    )
    target_test_loader = make_loader(
        spikes_tgt,
        behavior_tgt,
        labels_tgt,
        target_test_idx,
        target_model_session_id,
        cfg.adapt.batch_size,
        cfg.adapt.num_workers,
        shuffle=False,
    )

    source_train_loader_infinite = infinite_loader(source_train_loader)
    source_val_loader_infinite = infinite_loader(source_val_loader)
    source_train_iter = source_train_loader_infinite
    source_val_iter = source_val_loader_infinite

    shared_ae_config = build_shared_ae_config(cfg.model, cfg.dataset.signal_length)
    source_reference_ae = build_countwrapped_ae(
        all_session_configs,
        [source_session_id],
        shared_ae_config,
    )
    target_ae = build_countwrapped_ae(
        all_session_configs,
        [source_session_id, target_model_session_id],
        shared_ae_config,
    )
    behavior_head = BehaviorHead(cfg.model.C_latent)

    source_reference_ae.load_state_dict(ckpt["ae_state_dict"], strict=True)
    allowed_missing = [
        f"ae_net.read_in_layers.{target_model_session_id}.",
        f"ae_net.read_out_layers.{target_model_session_id}.",
    ]
    load_ae_state(target_ae, ckpt["ae_state_dict"], allowed_missing_prefixes=allowed_missing)
    behavior_head.load_state_dict(ckpt["behavior_head_state_dict"], strict=True)

    reset_modules = []
    if cfg.adapt.get("reset_target_read_adapter", False):
        reset_modules = reset_target_read_adapter(target_ae, target_model_session_id)
        print("Reset target read adapter modules from scratch:")
        for name in reset_modules:
            print(f"  {name}")

    source_reference_ae.to(device).eval()
    target_ae.to(device)
    behavior_head.to(device).eval()

    for param in source_reference_ae.parameters():
        param.requires_grad = False
    for param in behavior_head.parameters():
        param.requires_grad = False

    adaptation_params = freeze_all_but_read_adapters(target_ae, target_model_session_id)
    trainable_names = [name for name, p in target_ae.named_parameters() if p.requires_grad]
    print("Trainable target-adaptation parameters:")
    for name in trainable_names:
        print(f"  {name}")

    optimizer = torch.optim.AdamW(adaptation_params, lr=cfg.adapt.lr)
    criterion_poisson = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=len(target_train_loader) * cfg.adapt.num_warmup_epochs,
        num_training_steps=len(target_train_loader) * cfg.adapt.num_epochs,
    )

    train_history = []
    val_history = []
    pbar = tqdm(range(cfg.adapt.num_epochs), desc="Target adaptation with conditional MMD")

    for epoch in pbar:
        target_ae.train()
        source_reference_ae.eval()
        behavior_head.eval()
        epoch_train = {"recon": 0.0, "mmd": 0.0, "total": 0.0}

        for target_batch in target_train_loader:
            source_batch = next(source_train_iter)
            signal_tgt = target_batch["signal"].to(device)
            signal_src = source_batch["signal"].to(device)
            labels_tgt_batch = target_batch["label"].to(device)
            labels_src_batch = source_batch["label"].to(device)

            optimizer.zero_grad()

            input_signal_tgt, unmasked = apply_notebook_mask(signal_tgt, cfg.adapt.mask_prob)
            output_rates_tgt, z_target = target_ae(input_signal_tgt, target_model_session_id)
            recon_loss = (criterion_poisson(output_rates_tgt, signal_tgt) * unmasked).mean()

            with torch.no_grad():
                _, z_source = source_reference_ae(signal_src, source_session_id)

            mmd_loss = conditional_mmd_loss(
                z_source,
                labels_src_batch,
                z_target,
                labels_tgt_batch,
                device=device,
                kernel_mul=cfg.adapt.kernel_mul,
                kernel_num=cfg.adapt.kernel_num,
                max_samples_per_condition=cfg.adapt.max_mmd_samples_per_condition,
            )
            total_mmd_loss = cfg.adapt.eta_mmd * mmd_loss
            total_loss = recon_loss + total_mmd_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adaptation_params, cfg.adapt.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            batch_size = signal_tgt.size(0)
            epoch_train["recon"] += recon_loss.item() * batch_size
            epoch_train["mmd"] += total_mmd_loss.item() * batch_size
            epoch_train["total"] += total_loss.item() * batch_size

        avg_train = {k: v / len(target_train_loader.dataset) for k, v in epoch_train.items()}

        target_ae.eval()
        epoch_val = {"recon": 0.0, "mmd": 0.0, "total": 0.0}
        with torch.no_grad():
            for target_batch in target_val_loader:
                source_batch = next(source_val_iter)
                signal_tgt = target_batch["signal"].to(device)
                signal_src = source_batch["signal"].to(device)
                labels_tgt_batch = target_batch["label"].to(device)
                labels_src_batch = source_batch["label"].to(device)

                output_rates_tgt, z_target = target_ae(signal_tgt, target_model_session_id)
                recon_loss = criterion_poisson(output_rates_tgt, signal_tgt).mean()
                _, z_source = source_reference_ae(signal_src, source_session_id)

                mmd_loss = conditional_mmd_loss(
                    z_source,
                    labels_src_batch,
                    z_target,
                    labels_tgt_batch,
                    device=device,
                    kernel_mul=cfg.adapt.kernel_mul,
                    kernel_num=cfg.adapt.kernel_num,
                    max_samples_per_condition=cfg.adapt.max_mmd_samples_per_condition,
                )
                total_mmd_loss = cfg.adapt.eta_mmd * mmd_loss
                total_loss = recon_loss + total_mmd_loss

                batch_size = signal_tgt.size(0)
                epoch_val["recon"] += recon_loss.item() * batch_size
                epoch_val["mmd"] += total_mmd_loss.item() * batch_size
                epoch_val["total"] += total_loss.item() * batch_size

        avg_val = {k: v / len(target_val_loader.dataset) for k, v in epoch_val.items()}
        train_history.append(avg_train)
        val_history.append(avg_val)
        pbar.set_postfix({
            "train_total": f"{avg_train['total']:.4f}",
            "val_total": f"{avg_val['total']:.4f}",
            "train_mmd": f"{avg_train['mmd']:.4f}",
            "val_mmd": f"{avg_val['mmd']:.4f}",
        })

    r2_result, y_true_trials, y_pred_trials = evaluate_behavior_r2(
        target_ae,
        behavior_head,
        target_test_loader,
        target_model_session_id,
        device,
    )

    print("\n================ Conditional MMD Target Test Behavior R2 ================")
    print(r2_result)

    final_model_path = os.path.join(cfg.exp.save_dir, "target_adapt_conditional_mmd_model_final.pt")
    torch.save(
        {
            "target_session_id": target_session_id,
            "target_model_session_id": target_model_session_id,
            "source_session_id": source_session_id,
            "ae_state_dict": target_ae.state_dict(),
            "behavior_head_state_dict": behavior_head.state_dict(),
            "source_behavior_checkpoint": cfg.exp.source_behavior_checkpoint,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "target_train_indices": target_train_idx,
            "target_val_indices": target_val_idx,
            "target_test_indices": target_test_idx,
            "behavior_normalized": True,
            "source_behavior_mean": source_behavior_mean,
            "source_behavior_var": source_behavior_var,
            "source_behavior_std": source_behavior_std,
            "target_behavior_mean": target_behavior_mean,
            "target_behavior_var": target_behavior_var,
            "target_behavior_std": target_behavior_std,
            "reset_target_read_adapter": bool(cfg.adapt.get("reset_target_read_adapter", False)),
            "reset_target_read_adapter_modules": reset_modules,
            "train_history": train_history,
            "val_history": val_history,
            "test_r2": r2_result,
        },
        final_model_path,
    )
    print(f"Saved target adapted checkpoint to: {final_model_path}")

    loss_path = os.path.join(cfg.exp.loss_dir, "loss_curve_conditional_mmd.png")
    plot_loss_curve(train_history, val_history, loss_path)
    print(f"Saved loss curve to: {loss_path}")

    npz_path = os.path.join(cfg.exp.eval_dir, "target_test_behavior_prediction_conditional_mmd.npz")
    np.savez(
        npz_path,
        y_true=y_true_trials,
        y_pred=y_pred_trials,
        target_test_indices=target_test_idx,
        behavior_normalized=True,
        target_behavior_mean=target_behavior_mean,
        target_behavior_var=target_behavior_var,
        target_behavior_std=target_behavior_std,
        **r2_result,
    )
    print(f"Saved target test predictions to: {npz_path}")

    r2_csv_path = os.path.join(cfg.exp.eval_dir, "target_test_r2_conditional_mmd.csv")
    with open(r2_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_session_id",
                "target_session_id",
                "target_model_session_id",
                "target_train_ratio",
                "target_val_ratio",
                "target_test_ratio",
                "num_target_train_trials",
                "num_target_val_trials",
                "num_target_test_trials",
                "r2_x",
                "r2_y",
                "r2_uniform",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "source_session_id": source_session_id,
            "target_session_id": target_session_id,
            "target_model_session_id": target_model_session_id,
            "target_train_ratio": 0.2,
            "target_val_ratio": 0.2,
            "target_test_ratio": 0.6,
            "num_target_train_trials": len(target_train_idx),
            "num_target_val_trials": len(target_val_idx),
            "num_target_test_trials": len(target_test_idx),
            **r2_result,
        })
    print(f"Saved target test R2 CSV to: {r2_csv_path}")


if __name__ == "__main__":
    run_target_adaptation_conditional_mmd(cfg, TARGET_SESSION, SOURCE_SESSION)
