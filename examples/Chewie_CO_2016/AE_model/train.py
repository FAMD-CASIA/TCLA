import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import yaml
import accelerate
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import csv

from TCLA.data.data_multisession import get_session_data_and_configs, SingleSessionDataset
from TCLA.networks.count_wrapper import CountWrapper
from TCLA.networks.blocks import MultiSessionAutoEncoder
from TCLA.losses import latent_dynamics_regularizer


# ==============================================================================
#                      --- 1. 全局配置 ---
# ==============================================================================

SOURCE_SESSION = os.environ.get("SOURCE_SESSION_ID", "session_0")


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


LATENT_L2_BETA = env_float("LATENT_L2_BETA", 0.01)
LATENT_CURVATURE_BETA = env_float("LATENT_CURVATURE_BETA", 0.0)
LAMBDA_BEHAVIOR_CURV = env_float("LAMBDA_BEHAVIOR_CURV", 100000.0)
LATENT_SMOOTH_SIGMA_BINS = env_float("LATENT_SMOOTH_SIGMA_BINS", 10.0)
MASK_PROB = env_float("MASK_PROB", 0.5)
LAMBDA_BEHAVIOR = env_float("LAMBDA_BEHAVIOR", 1.0)
NUM_EPOCHS = env_int("NUM_EPOCHS", 400)


def format_tag(value):
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.0e}" if abs(value) < 1e-3 or abs(value) >= 1e3 else str(value).replace(".", "p")


RUN_TAG = (
    f"sigma{format_tag(LATENT_SMOOTH_SIGMA_BINS)}"
    f"_mp{format_tag(MASK_PROB)}"
    f"_ep{NUM_EPOCHS}"
    f"_ll2{format_tag(LATENT_L2_BETA)}"
    f"_lcurv{format_tag(LATENT_CURVATURE_BETA)}"
    f"_behcurv{format_tag(LAMBDA_BEHAVIOR_CURV)}"
)

cfg_yaml = f"""
# ----------------- 模型架构参数 -----------------
model:
  C_shared: 30
  C: 256
  C_latent: 16
  num_blocks: 4
  num_blocks_decoder: 0
  num_lin_per_mlp: 2
  bidirectional: False
  latent_smooth_sigma_bins: {LATENT_SMOOTH_SIGMA_BINS}
  use_pnba_read_adapter: True
  # pnba_proj_channels: PNBA卷积投影后的通道数 C_proj
  # pnba_pooled_neurons: 自适应池化后的共享“神经元槽位”数 D
  # 共享输入维度 = C_proj * D
  pnba_proj_channels: 32
  pnba_pooled_neurons: 16

# ----------------- 数据集参数 -----------------
dataset:
  datapath_root: {REPO_ROOT}/data/Chewie_CO_2016
  signal_length: 210

# ----------------- 训练参数 -----------------
training:
  lr: 0.001
  num_epochs: {NUM_EPOCHS}
  num_warmup_epochs: 100
  batch_size: 64
  num_workers: 4
  precision: "no"
  grad_clip: 1.0
  random_seed: 42
  latent_l2_beta: {LATENT_L2_BETA}
  latent_curvature_beta: {LATENT_CURVATURE_BETA}
  mask_prob: {MASK_PROB}

  # 行为学辅助预测损失权重
  lambda_behavior: {LAMBDA_BEHAVIOR}
  lambda_behavior_curv: {LAMBDA_BEHAVIOR_CURV}

# ----------------- 实验和日志参数 -----------------
exp:
  save_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/AE_model/source_{SOURCE_SESSION}_direct_shift_experiment/model/
  loss_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/AE_model/source_{SOURCE_SESSION}_direct_shift_experiment/loss/
  eval_dir: {REPO_ROOT}/outputs/Chewie_CO_2016/AE_model/source_{SOURCE_SESSION}_direct_shift_experiment/eval/
  run_direct_cross_session_eval: False
"""

config = OmegaConf.create(yaml.safe_load(cfg_yaml))
config.training.num_warmup_epochs = env_int("NUM_WARMUP_EPOCHS", config.training.num_warmup_epochs)
config.training.batch_size = env_int("BATCH_SIZE", config.training.batch_size)
config.training.num_workers = env_int("NUM_WORKERS", config.training.num_workers)


# ==============================================================================
#                      --- 2. 行为预测头 ---
# ==============================================================================

class BehaviorHead(nn.Module):
    """
    从 latent trajectory z 中预测行为学变量。

    输入:
        z: [B, C_latent, T]

    输出:
        behavior_pred: [B, 2, T]
    """
    def __init__(self, C_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C_latent, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 2, kernel_size=1)
        )

    def forward(self, z):
        return self.net(z)


def prepare_behavior_target(behavior, behavior_pred):
    """
    将 batch['behavior'] 整理成和 behavior_pred 一致的形状 [B, 2, T]。

    兼容:
        behavior: [B, 2, T]
        behavior: [B, T, 2]

    如果时间长度和 latent 输出长度不一致，则用线性插值对齐到 behavior_pred 的 T。
    """
    # behavior = behavior.float()

    # 如果是 [B, T, 2]，转成 [B, 2, T]
    if behavior.shape[1] != 2 and behavior.shape[2] == 2:
        behavior = behavior.permute(0, 2, 1).contiguous()

    return behavior


def behavior_curvature_regularizer(behavior_pred, behavior_target):
    """Second-order loss on predicted behavior curvature.

    This encourages the behavior head output to have trajectory curvature
    similar to the target behavior, without directly smoothing the target.
    """
    if behavior_pred.shape[-1] < 3:
        return torch.tensor(0.0, device=behavior_pred.device, dtype=behavior_pred.dtype)
    pred_curv = behavior_pred[:, :, 2:] - 2.0 * behavior_pred[:, :, 1:-1] + behavior_pred[:, :, :-2]
    target_curv = behavior_target[:, :, 2:] - 2.0 * behavior_target[:, :, 1:-1] + behavior_target[:, :, :-2]
    return F.mse_loss(pred_curv, target_curv, reduction='mean')


def compute_behavior_zscore_stats(behavior, train_indices, eps=1e-6):
    """
    Estimate behavior normalization only from the given session training split.

    behavior shape:
        trials x time x dims

    Returns mean/variance/std with shape [1, 1, dims], suitable for broadcasting.
    """
    train_behavior = behavior[train_indices]
    behavior_mean = train_behavior.mean(axis=(0, 1), keepdims=True)
    behavior_var = train_behavior.var(axis=(0, 1), keepdims=True)
    behavior_std = np.sqrt(np.maximum(behavior_var, eps))
    return behavior_mean, behavior_var, behavior_std


def normalize_behavior(behavior, behavior_mean, behavior_std):
    return ((behavior - behavior_mean) / behavior_std).astype(np.float32)


# ==============================================================================
#                      --- 3. 绘图函数 ---
# ==============================================================================

def plot_and_save_losses(train_losses, val_losses, epoch, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    epochs = range(1, epoch + 2)

    train_total = [t['total'] for t in train_losses]
    val_total = [v['total'] for v in val_losses]

    train_recon = [t['recon'] for t in train_losses]
    val_recon = [v['recon'] for v in val_losses]

    train_latent = [t['latent'] for t in train_losses]
    val_latent = [v['latent'] for v in val_losses]

    train_behavior = [t['behavior'] for t in train_losses]
    val_behavior = [v['behavior'] for v in val_losses]

    train_behavior_curv = [t.get('behavior_curv', 0.0) for t in train_losses]
    val_behavior_curv = [v.get('behavior_curv', 0.0) for v in val_losses]

    axes[0].plot(epochs, train_total, label='Train Total')
    axes[0].plot(epochs, val_total, label='Val Total', linestyle='--')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_recon, label='Train Recon')
    axes[1].plot(epochs, val_recon, label='Val Recon', linestyle='--')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Poisson NLL')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, train_latent, label='Train Latent')
    axes[2].plot(epochs, val_latent, label='Val Latent', linestyle='--')
    axes[2].set_title('Weighted Latent Regularization')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(epochs, train_behavior, label='Train Behavior')
    axes[3].plot(epochs, val_behavior, label='Val Behavior', linestyle='--')
    axes[3].set_title('Weighted Behavior Loss')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('MSE')
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(epochs, train_behavior_curv, label='Train BehCurv')
    axes[4].plot(epochs, val_behavior_curv, label='Val BehCurv', linestyle='--')
    axes[4].set_title('Weighted Behavior Curvature Loss')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Curvature MSE')
    axes[4].legend()
    axes[4].grid(True)

    fig.suptitle(f'Losses up to Epoch {epoch + 1}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path)
    plt.close(fig)


def get_session_ids_from_datapath(datapath_root):
    session_ids = []
    for filename in os.listdir(datapath_root):
        if not filename.endswith(".pickle"):
            continue
        stem = filename[:-7]
        if not stem.startswith("session_"):
            continue
        session_ids.append(stem)
    session_ids.sort(key=lambda x: int(x.split("_")[-1]))
    return session_ids


def plot_trajectory_comparison(y_true_trials, y_pred_trials, save_path, max_trials=60):
    n_trials = y_true_trials.shape[0]
    draw_trials = min(n_trials, max_trials)
    fig, ax = plt.subplots(figsize=(8, 8))
    for trial_id in range(draw_trials):
        true_label = "True trajectory" if trial_id == 0 else None
        pred_label = "Predicted trajectory" if trial_id == 0 else None
        ax.plot(
            y_true_trials[trial_id, 0, :],
            y_true_trials[trial_id, 1, :],
            color="#1f77b4",
            linewidth=1.1,
            alpha=0.75,
            label=true_label,
        )
        ax.plot(
            y_pred_trials[trial_id, 0, :],
            y_pred_trials[trial_id, 1, :],
            color="#d62728",
            linewidth=1.1,
            alpha=0.75,
            label=pred_label,
        )
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Trajectory comparison (first {draw_trials}/{n_trials} trials)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def split_target_20_20_60(labels, seed):
    indices = np.arange(len(labels))
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


def run_direct_cross_session_eval(
    cfg,
    source_session_id,
    source_test_indices,
    model,
    behavior_head,
    device,
    source_behavior_mean,
    source_behavior_std,
):
    eval_root = cfg.exp.eval_dir
    os.makedirs(eval_root, exist_ok=True)

    session_ids = get_session_ids_from_datapath(cfg.dataset.datapath_root)
    all_session_data, _ = get_session_data_and_configs(session_ids, cfg.dataset.datapath_root)

    summary_rows = []
    model.eval()
    behavior_head.eval()
    for session_id in session_ids:
        session_data = all_session_data[session_id]
        spikes = session_data["spikes"]
        raw_behavior = session_data["behavior"]
        labels = session_data["label"]
        if session_id == source_session_id:
            eval_indices = np.asarray(source_test_indices)
            eval_mode = "test_split"
            target_train_indices = np.asarray([], dtype=int)
            target_val_indices = np.asarray([], dtype=int)
            target_test_indices = np.asarray([], dtype=int)
            session_behavior_mean = source_behavior_mean
            session_behavior_var = np.square(source_behavior_std)
            session_behavior_std = source_behavior_std
        else:
            target_train_indices, target_val_indices, target_test_indices = split_target_20_20_60(
                labels,
                seed=cfg.training.random_seed,
            )
            eval_indices = target_test_indices
            eval_mode = "target_test_60pct"
            session_behavior_mean, session_behavior_var, session_behavior_std = compute_behavior_zscore_stats(
                raw_behavior,
                target_train_indices,
            )

        behavior = normalize_behavior(
            raw_behavior,
            session_behavior_mean,
            session_behavior_std,
        )

        eval_dataset = SingleSessionDataset(
            spikes[eval_indices],
            behavior[eval_indices],
            labels[eval_indices],
            session_id,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )

        true_list = []
        pred_list = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Direct eval on {session_id} ({eval_mode})"):
                signal = batch["signal"].to(device)
                _, z = model(signal, session_id)
                behavior_pred = behavior_head(z)
                behavior_true = prepare_behavior_target(batch["behavior"].to(device), behavior_pred)

                true_list.append(behavior_true.cpu())
                pred_list.append(behavior_pred.cpu())

        y_true_trials = torch.cat(true_list, dim=0)  # [B, 2, T]
        y_pred_trials = torch.cat(pred_list, dim=0)  # [B, 2, T]
        y_true_flat = y_true_trials.permute(0, 2, 1).reshape(-1, 2).numpy()
        y_pred_flat = y_pred_trials.permute(0, 2, 1).reshape(-1, 2).numpy()

        r2_xy = r2_score(y_true_flat, y_pred_flat, multioutput="raw_values")
        mean_r2 = float(np.mean(r2_xy))
        session_num = int(session_id.split("_")[-1])

        session_dir = os.path.join(eval_root, session_id)
        os.makedirs(session_dir, exist_ok=True)
        np.savez(
            os.path.join(session_dir, "behavior_prediction_direct_eval.npz"),
            y_true=y_true_trials.numpy(),
            y_pred=y_pred_trials.numpy(),
            y_true_flat=y_true_flat,
            y_pred_flat=y_pred_flat,
            eval_indices=eval_indices,
            eval_mode=eval_mode,
            target_train_indices=target_train_indices,
            target_val_indices=target_val_indices,
            target_test_indices=target_test_indices,
            r2_x=float(r2_xy[0]),
            r2_y=float(r2_xy[1]),
            mean_r2=mean_r2,
            behavior_normalized=True,
            behavior_mean=session_behavior_mean,
            behavior_var=session_behavior_var,
            behavior_std=session_behavior_std,
        )
        plot_trajectory_comparison(
            y_true_trials.numpy(),
            y_pred_trials.numpy(),
            os.path.join(session_dir, f"trajectory_comparison_on_{session_id}.svg"),
        )

        summary_rows.append({
            "session_id": session_id,
            "session_num": session_num,
            "eval_mode": eval_mode,
            "num_eval_trials": int(len(eval_indices)),
            "r2_x": float(r2_xy[0]),
            "r2_y": float(r2_xy[1]),
            "mean_r2": mean_r2,
        })
        print(
            f"[Direct eval] {session_id} ({eval_mode}, n={len(eval_indices)}) | "
            f"R2_x={r2_xy[0]:.4f}, "
            f"R2_y={r2_xy[1]:.4f}, mean_R2={mean_r2:.4f}"
        )

    summary_rows.sort(key=lambda x: x["session_num"])
    summary_csv = os.path.join(eval_root, "direct_cross_session_r2_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session_id", "session_num", "eval_mode", "num_eval_trials", "r2_x", "r2_y", "mean_r2"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if summary_rows:
        x = [row["session_num"] for row in summary_rows]
        y = [row["mean_r2"] for row in summary_rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, marker="o", linewidth=2, color="#d62728")
        ax.set_xlabel("Session ID")
        ax.set_ylabel("Mean R2")
        ax.set_title("Direct Cross-Session Decoding Without Adaptation")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        trend_path = os.path.join(eval_root, "direct_cross_session_r2_trend.svg")
        fig.savefig(trend_path)
        plt.close(fig)
        print(f"Direct cross-session trend saved to: {trend_path}")

    print(f"Direct cross-session summary saved to: {summary_csv}")


# ==============================================================================
#                            --- 4. 主训练函数 ---
# ==============================================================================

def train_main_ae_with_behavior(cfg, source_session_id):
    """
    Train the source-session AE with smoothed latents and an auxiliary behavior head.

    Total loss:
        total_loss =
            reconstruction_loss
            + latent_dynamics_regularizer(z, cfg)
            + lambda_behavior * behavior_loss
            + lambda_behavior_curv * behavior_curvature_loss

    Here ``latent_dynamics_regularizer`` is already weighted internally by
    ``latent_l2_beta`` and ``latent_curvature_beta``. No extra outer scaling is
    applied in the training loop.
    """

    torch.manual_seed(cfg.training.random_seed)
    np.random.seed(cfg.training.random_seed)

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.training.precision,
    )
    device = accelerator.device

    # --------------------------------------------------------------------------
    # Step 1: 加载 source session 数据
    # --------------------------------------------------------------------------
    all_session_data, session_configs = get_session_data_and_configs(
        [source_session_id],
        datapath_root=cfg.dataset.datapath_root
    )

    source_data = all_session_data[source_session_id]
    spikes = source_data['spikes']
    behavior = source_data['behavior']
    labels = source_data['label']

    indices = np.arange(len(spikes))

    # 8:1:1 stratified split
    train_indices, temp_test_indices, train_labels, _ = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=cfg.training.random_seed,
        stratify=labels
    )

    val_indices, test_indices, _, _ = train_test_split(
        temp_test_indices,
        labels[temp_test_indices],
        test_size=0.5,
        random_state=cfg.training.random_seed,
        stratify=labels[temp_test_indices]
    )

    print(f"--- Data Split for Source Session '{source_session_id}' (Stratified 8:1:1) ---")
    print(f"Total trials: {len(spikes)}")
    print(f"  - Train set size: {len(train_indices)}")
    print(f"  - Validation set size: {len(val_indices)}")
    print(f"  - Test set size: {len(test_indices)}")

    behavior_mean, behavior_var, behavior_std = compute_behavior_zscore_stats(
        behavior,
        train_indices,
    )
    behavior = normalize_behavior(behavior, behavior_mean, behavior_std)
    cfg.dataset.behavior_normalized = True
    cfg.dataset.behavior_mean = behavior_mean.reshape(-1).tolist()
    cfg.dataset.behavior_var = behavior_var.reshape(-1).tolist()
    cfg.dataset.behavior_std = behavior_std.reshape(-1).tolist()
    print("--- Behavior z-score normalization from source train split ---")
    print(f"  - mean: {cfg.dataset.behavior_mean}")
    print(f"  - var:  {cfg.dataset.behavior_var}")
    print(f"  - std:  {cfg.dataset.behavior_std}")
    print("  - spikes/rates are not normalized; Poisson reconstruction loss is unchanged.")

    train_dataset = SingleSessionDataset(
        spikes[train_indices],
        behavior[train_indices],
        labels[train_indices],
        source_session_id
    )

    val_dataset = SingleSessionDataset(
        spikes[val_indices],
        behavior[val_indices],
        labels[val_indices],
        source_session_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )

    # --------------------------------------------------------------------------
    # Step 2: 初始化 AE 和 behavior head
    # --------------------------------------------------------------------------
    train_session_configs = {
        source_session_id: session_configs[source_session_id]
    }

    shared_ae_config = {
        'C_in': cfg.model.C_shared,
        'C': cfg.model.C,
        'C_latent': cfg.model.C_latent,
        'L': cfg.dataset.signal_length,
        'num_blocks': cfg.model.num_blocks,
        'bidirectional': cfg.model.bidirectional,
        'num_blocks_decoder': cfg.model.num_blocks_decoder,
        'num_lin_per_mlp': cfg.model.num_lin_per_mlp,
        'latent_smooth_sigma_bins': cfg.model.latent_smooth_sigma_bins,
        'use_pnba_read_adapter': cfg.model.use_pnba_read_adapter,
        'pnba_proj_channels': cfg.model.pnba_proj_channels,
        'pnba_pooled_neurons': cfg.model.pnba_pooled_neurons,
    }

    model = MultiSessionAutoEncoder(train_session_configs, shared_ae_config)
    model = CountWrapper(model)

    behavior_head = BehaviorHead(cfg.model.C_latent)

    num_ae_params = sum(p.numel() for p in model.parameters())
    num_behavior_params = sum(p.numel() for p in behavior_head.parameters())

    print(f"Initialized MultiSessionAutoEncoder with {num_ae_params / 1e6:.2f}M parameters.")
    print(f"Initialized BehaviorHead with {num_behavior_params / 1e3:.2f}K parameters.")
    print(f"lambda_behavior = {cfg.training.lambda_behavior}")
    print(
        "PNBA adapter setting | "
        f"use={cfg.model.use_pnba_read_adapter}, "
        f"proj_channels={cfg.model.pnba_proj_channels}, "
        f"pooled_neurons={cfg.model.pnba_pooled_neurons}, "
        f"shared_input_dim={cfg.model.pnba_proj_channels * cfg.model.pnba_pooled_neurons}"
    )

    # --------------------------------------------------------------------------
    # Step 3: optimizer, scheduler, losses
    # --------------------------------------------------------------------------
    params = list(model.parameters()) + list(behavior_head.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.training.lr
    )

    criterion_poisson = nn.PoissonNLLLoss(
        log_input=False,
        full=True,
        reduction="none"
    )

    num_training_steps = len(train_loader) * cfg.training.num_epochs

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.training.num_warmup_epochs * len(train_loader),
        num_training_steps=num_training_steps,
    )

    model, behavior_head, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,
        behavior_head,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler
    )

    # --------------------------------------------------------------------------
    # Step 4: 保存路径
    # --------------------------------------------------------------------------
    save_dir = cfg.exp.save_dir
    os.makedirs(save_dir, exist_ok=True)

    save_loss_dir = cfg.exp.loss_dir
    os.makedirs(save_loss_dir, exist_ok=True)

    split_info_path = os.path.join(save_dir, "source_split_indices.pkl")
    if accelerator.is_local_main_process:
        with open(split_info_path, "wb") as f:
            pickle.dump(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                    "source_session_id": source_session_id,
                    "behavior_normalized": True,
                    "behavior_mean": behavior_mean,
                    "behavior_var": behavior_var,
                    "behavior_std": behavior_std,
                },
                f
            )
        print(f"Saved split indices to: {split_info_path}")

    train_loss_history = []
    val_loss_history = []

    # --------------------------------------------------------------------------
    # Step 5: 训练循环
    # --------------------------------------------------------------------------
    pbar = tqdm(
        range(cfg.training.num_epochs),
        disable=not accelerator.is_local_main_process,
        desc="Training Source AE + Behavior Head"
    )

    for epoch in pbar:
        model.train()
        behavior_head.train()

        epoch_train_losses = {
            'recon': 0.0,
            'latent': 0.0,
            'behavior': 0.0,
            'behavior_curv': 0.0,
            'total': 0.0
        }

        for batch in train_loader:
            signal = batch['signal']

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # coordinated dropout / masking (same logic as notebook version)
                mask_prob = cfg.training.get("mask_prob", 0.5)
                if mask_prob < 0 or mask_prob >= 1:
                    raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}")

                if mask_prob > 0:
                    mask = (torch.rand_like(signal) > mask_prob).float()
                    input_signal = signal * (mask / (1 - mask_prob))
                    unmasked = 1 - mask
                else:
                    mask = torch.ones_like(signal)
                    input_signal = signal
                    unmasked = torch.ones_like(signal)

                # AE forward
                output_rates, z = model(input_signal, source_session_id)

                # behavior prediction from latent
                behavior_pred = behavior_head(z)
                behavior_target = prepare_behavior_target(
                    batch['behavior'],
                    behavior_pred
                )

                # loss
                poisson_loss = criterion_poisson(output_rates, signal) * unmasked
                poisson_loss = poisson_loss.mean()
                recon_loss = poisson_loss

                weighted_latent_loss = latent_dynamics_regularizer(z, cfg)

                behavior_loss = F.mse_loss(
                    behavior_pred,
                    behavior_target,
                    reduction='mean',
                )
                behavior_curv_loss = behavior_curvature_regularizer(
                    behavior_pred,
                    behavior_target,
                )

                weighted_behavior_loss = cfg.training.lambda_behavior * behavior_loss
                weighted_behavior_curv_loss = (
                    cfg.training.lambda_behavior_curv * behavior_curv_loss
                )

                total_loss = (
                    recon_loss
                    + weighted_latent_loss
                    + weighted_behavior_loss
                    + weighted_behavior_curv_loss
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        params,
                        cfg.training.grad_clip
                    )

                optimizer.step()
                lr_scheduler.step()

                batch_size = signal.size(0)

                epoch_train_losses['recon'] += recon_loss.item() * batch_size
                epoch_train_losses['latent'] += weighted_latent_loss.item() * batch_size
                epoch_train_losses['behavior'] += weighted_behavior_loss.item() * batch_size
                epoch_train_losses['behavior_curv'] += weighted_behavior_curv_loss.item() * batch_size
                epoch_train_losses['total'] += total_loss.item() * batch_size

        # ----------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------
        model.eval()
        behavior_head.eval()

        epoch_val_losses = {
            'recon': 0.0,
            'latent': 0.0,
            'behavior': 0.0,
            'behavior_curv': 0.0,
            'total': 0.0
        }

        with torch.no_grad():
            for batch in val_loader:
                signal = batch['signal']

                output_rates, z = model(signal, source_session_id)

                behavior_pred = behavior_head(z)
                behavior_target = prepare_behavior_target(
                    batch['behavior'],
                    behavior_pred
                )

                recon_loss = criterion_poisson(output_rates, signal).mean()

                weighted_latent_loss = latent_dynamics_regularizer(z, cfg)

                behavior_loss = F.mse_loss(
                    behavior_pred,
                    behavior_target,
                    reduction='mean',
                )
                behavior_curv_loss = behavior_curvature_regularizer(
                    behavior_pred,
                    behavior_target,
                )

                weighted_behavior_loss = cfg.training.lambda_behavior * behavior_loss
                weighted_behavior_curv_loss = (
                    cfg.training.lambda_behavior_curv * behavior_curv_loss
                )

                total_loss = (
                    recon_loss
                    + weighted_latent_loss
                    + weighted_behavior_loss
                    + weighted_behavior_curv_loss
                )

                batch_size = signal.size(0)

                epoch_val_losses['recon'] += recon_loss.item() * batch_size
                epoch_val_losses['latent'] += weighted_latent_loss.item() * batch_size
                epoch_val_losses['behavior'] += weighted_behavior_loss.item() * batch_size
                epoch_val_losses['behavior_curv'] += weighted_behavior_curv_loss.item() * batch_size
                epoch_val_losses['total'] += total_loss.item() * batch_size

        avg_train_losses = {
            k: v / len(train_loader.dataset)
            for k, v in epoch_train_losses.items()
        }

        avg_val_losses = {
            k: v / len(val_loader.dataset)
            for k, v in epoch_val_losses.items()
        }

        train_loss_history.append(avg_train_losses)
        val_loss_history.append(avg_val_losses)

        pbar.set_postfix(
            {
                "Train Total": f"{avg_train_losses['total']:.4f}",
                "Val Total": f"{avg_val_losses['total']:.4f}",
                "Recon": f"{avg_val_losses['recon']:.4f}",
                "Beh": f"{avg_val_losses['behavior']:.4f}",
                "BehCurv": f"{avg_val_losses['behavior_curv']:.4f}",
            }
        )

        # ----------------------------------------------------------------------
        # Save checkpoints
        # ----------------------------------------------------------------------
        if accelerator.is_local_main_process and (epoch + 1) % 20 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_behavior_head = accelerator.unwrap_model(behavior_head)

            # 1. 保存 AE-only state_dict：兼容你后续 Stage 2 alignment 代码
            ae_only_path = os.path.join(
                save_dir,
                f"full_ae_model.pt"
            )
            torch.save(
                unwrapped_model.state_dict(),
                ae_only_path
            )

            # 2. 保存完整 checkpoint：包含 behavior head
            full_ckpt_path = os.path.join(
                save_dir,
                f"full_ae_behavior_model.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "source_session_id": source_session_id,
                    "ae_state_dict": unwrapped_model.state_dict(),
                    "behavior_head_state_dict": unwrapped_behavior_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                    "behavior_normalized": True,
                    "behavior_mean": behavior_mean,
                    "behavior_var": behavior_var,
                    "behavior_std": behavior_std,
                },
                full_ckpt_path
            )

            print(f"\nAE-only model saved to: {ae_only_path}")
            print(f"Full AE + behavior checkpoint saved to: {full_ckpt_path}")

            loss_plot_path = os.path.join(
                save_loss_dir,
                "loss_curve.png"
            )
            plot_and_save_losses(
                train_loss_history,
                val_loss_history,
                epoch,
                loss_plot_path
            )

            history_path = os.path.join(
                save_loss_dir,
                "loss_history.pkl"
            )
            with open(history_path, "wb") as f:
                pickle.dump(
                    {
                        "train_loss_history": train_loss_history,
                        "val_loss_history": val_loss_history,
                    },
                    f
                )

    # --------------------------------------------------------------------------
    # Save final model
    # --------------------------------------------------------------------------
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_behavior_head = accelerator.unwrap_model(behavior_head)

        final_ae_path = os.path.join(save_dir, "full_ae_model_final.pt")
        final_full_path = os.path.join(save_dir, "full_ae_behavior_model_final.pt")

        torch.save(
            unwrapped_model.state_dict(),
            final_ae_path
        )

        torch.save(
            {
                "epoch": cfg.training.num_epochs,
                "source_session_id": source_session_id,
                "ae_state_dict": unwrapped_model.state_dict(),
                "behavior_head_state_dict": unwrapped_behavior_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "behavior_normalized": True,
                "behavior_mean": behavior_mean,
                "behavior_var": behavior_var,
                "behavior_std": behavior_std,
            },
            final_full_path
        )

        print(f"\nFinal AE-only model saved to: {final_ae_path}")
        print(f"Final AE + behavior checkpoint saved to: {final_full_path}")

        if cfg.exp.get("run_direct_cross_session_eval", True):
            run_direct_cross_session_eval(
                cfg=cfg,
                source_session_id=source_session_id,
                source_test_indices=test_indices,
                model=unwrapped_model.to(device).eval(),
                behavior_head=unwrapped_behavior_head.to(device).eval(),
                device=device,
                source_behavior_mean=behavior_mean,
                source_behavior_std=behavior_std,
            )
        else:
            print("Skipping direct cross-session eval.")


# ==============================================================================
#                           --- 5. 脚本主入口 ---
# ==============================================================================

if __name__ == '__main__':
    print("===================================================")
    print("      STARTING AE + BEHAVIOR TRAINING")
    print("===================================================")

    train_main_ae_with_behavior(config, SOURCE_SESSION)
