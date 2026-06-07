import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
import pdb
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import yaml
import accelerate
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ldns.data.data_multisession import get_session_data_and_configs, SingleSessionDataset
from ldns.networks import CountWrapper, MultiSessionAutoEncoder
from ldns.losses import latent_regularizer

# ==============================================================================
#                      --- 1. 全局配置 ---
# ==============================================================================

left_out_session = 'session_0' 
SOURCE_SESSION = 'session_4'

# 将所有配置参数直接写在这里
cfg_yaml = """
# ----------------- 模型架构参数 -----------------
model:
  C_shared: 30
  C: 256
  C_latent: 16
  num_blocks: 4
  num_blocks_decoder: 0
  num_lin_per_mlp: 2
  bidirectional: False

# ----------------- 数据集参数 -----------------
dataset:
  datapath_root: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold_with_label/data
  signal_length: 320

# ----------------- 训练参数 -----------------
training:
  lr: 0.001
  num_epochs: 1000
  num_warmup_epochs: 100
  batch_size: 64
  num_workers: 4
  precision: "no"
  grad_clip: 1.0
  random_seed: 42
  td_k: 5
  latent_beta: 0.001
  latent_td_beta: 0.2
  mask_prob: 0.5

# ----------------- 实验和日志参数 -----------------
exp:
  save_dir: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold_with_label/exp/AE_model/model_train_on_4_left_0/model
  loss_dir: /data0/user/cyzhao/LD-AE_Multi_cross_n_fold_with_label/exp/AE_model/model_train_on_4_left_0/model_loss
"""

# 从YAML字符串创建OmegaConf对象
config = OmegaConf.create(yaml.safe_load(cfg_yaml))

# ==============================================================================
#                      --- 2. 绘图函数 ---
# ==============================================================================
def plot_and_save_losses(train_losses, val_losses, epoch, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, epoch + 2)
    
    train_total_losses = [t['total'] for t in train_losses]
    val_total_losses = [v['total'] for v in val_losses]
    
    ax.plot(epochs, train_total_losses, label='Train Total Loss', color='royalblue')
    ax.plot(epochs, val_total_losses, label='Validation Total Loss', color='darkorange', linestyle='--')
    
    ax.set_title('Train vs. Validation Loss Curve')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True)
    
    fig.suptitle(f'Losses up to Epoch {epoch+1}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path)
    plt.close(fig)


# ==============================================================================
#                            --- 3. 主训练函数 ---
# ==============================================================================

def train_main_ae(cfg, source_session_id, left_out_session_id):
    """
    Trains the shared AutoEncoder body .
    """

    torch.manual_seed(cfg.training.random_seed)
    np.random.seed(cfg.training.random_seed)
    
    # --- Setup Accelerator ---
    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.training.precision,
    )
    device = accelerator.device
    
    # --- Load Data for source sessions ---
    all_session_data, session_configs = get_session_data_and_configs(
        [source_session_id],
        datapath_root=cfg.dataset.datapath_root
    )
    source_data = all_session_data[source_session_id]
    spikes = source_data['spikes']
    behavior = source_data['behavior']
    labels = source_data['label']

    # 第一次划分：80% 训练 vs 20% 临时测试 (分层)
    indices = np.arange(len(spikes))
    train_indices, temp_test_indices, train_labels, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=cfg.training.random_seed, stratify=labels
    )

    # 第二次划分：从 20% 中分出 10% 验证 和 10% 测试 (分层)
    val_indices, test_indices, _, _ = train_test_split(
        temp_test_indices, labels[temp_test_indices], test_size=0.5, 
        random_state=cfg.training.random_seed, stratify=labels[temp_test_indices]
    )
    print(f"--- Data Split for Source Session '{source_session_id}' (Stratified 8:1:1) ---")
    print(f"Total trials: {len(spikes)}")
    print(f"  - Train set size: {len(train_indices)}")
    print(f"  - Validation set size: {len(val_indices)}")
    print(f"  - Test set size: {len(test_indices)}")
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

    # --- Initialize Model ---
    train_session_configs = {source_session_id: session_configs[source_session_id]}
    shared_ae_config = {
        'C_in': cfg.model.C_shared,
        'C': cfg.model.C,
        'C_latent': cfg.model.C_latent,
        'L': cfg.dataset.signal_length,
        'num_blocks': cfg.model.num_blocks,
        'bidirectional': cfg.model.bidirectional,
        'num_blocks_decoder': cfg.model.num_blocks_decoder,
        'num_lin_per_mlp': cfg.model.num_lin_per_mlp
    }
    
    model = MultiSessionAutoEncoder(train_session_configs, shared_ae_config)
    model = CountWrapper(model)
    
    print(f"Initialized MultiSessionAutoEncoder with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- Setup Optimizer, Scheduler, and Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    criterion_poisson = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")

    num_training_steps = len(train_loader) * cfg.training.num_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.training.num_warmup_epochs * len(train_loader),
        num_training_steps=num_training_steps,
    )

    # --- Prepare with Accelerator ---
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # 模型保存路径
    save_dir = cfg.exp.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_loss_dir = cfg.exp.loss_dir
    os.makedirs(save_loss_dir, exist_ok=True)
    
    train_loss_history, val_loss_history = [], []

    # --- Training Loop ---
    pbar = tqdm(range(cfg.training.num_epochs), disable=not accelerator.is_local_main_process, desc="Training Main AE")
    for epoch in pbar:
        model.train()

        # 为每个epoch重置loss
        epoch_train_losses = {'recon': 0.0, 'latent': 0.0, 'total': 0.0}

        for batch in train_loader:
            signal = batch['signal']
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                mask = (torch.rand_like(signal) > cfg.training.mask_prob).float()
                input_signal = signal * mask / (1 - cfg.training.mask_prob + 1e-8)
                output_rates, z = model(input_signal, source_session_id)
                
                recon_loss = criterion_poisson(output_rates, signal).mean()
                numel = signal.shape[0] * signal.shape[1] * signal.shape[2]
                latent_loss = latent_regularizer(z, cfg) / numel
                total_loss = recon_loss + cfg.training.latent_beta * latent_loss
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                
                batch_size = signal.size(0)
                epoch_train_losses['recon'] += recon_loss.item() * batch_size
                epoch_train_losses['latent'] += (cfg.training.latent_beta * latent_loss.item()) * batch_size
                epoch_train_losses['total'] += total_loss.item() * batch_size

        # --- 验证 ---
        model.eval()
        epoch_val_losses = {'recon': 0.0, 'latent': 0.0, 'total': 0.0}
        with torch.no_grad():
            for batch in val_loader:
                signal = batch['signal']
                output_rates, z = model(signal, source_session_id)
                recon_loss = criterion_poisson(output_rates, signal).mean()
                numel = signal.shape[0] * signal.shape[1] * signal.shape[2]
                latent_loss = latent_regularizer(z, cfg) / numel
                total_loss = recon_loss + cfg.training.latent_beta * latent_loss
                
                batch_size = signal.size(0)
                epoch_val_losses['recon'] += recon_loss.item() * batch_size
                epoch_val_losses['latent'] += (cfg.training.latent_beta * latent_loss.item()) * batch_size
                epoch_val_losses['total'] += total_loss.item() * batch_size

        # --- 记录与更新 ---
        avg_train_losses = {k: v / len(train_loader.dataset) for k, v in epoch_train_losses.items()}
        avg_val_losses = {k: v / len(val_loader.dataset) for k, v in epoch_val_losses.items()}
        train_loss_history.append(avg_train_losses); val_loss_history.append(avg_val_losses)
        
        pbar.set_postfix({"Train Loss": f"{avg_train_losses['total']:.4f}", "Val Loss": f"{avg_val_losses['total']:.4f}"})

        # 保存模型 绘制loss图
        if accelerator.is_local_main_process and (epoch + 1) % 20 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(save_dir, f"full_ae_model_{epoch + 1}.pt")
            torch.save(unwrapped_model.state_dict(), save_path)

            print(f"\nModel saved to {save_path} at epoch {epoch+1}")
            loss_plot_path = os.path.join(save_loss_dir, f"loss_curve_epoch_{epoch+1}.png")
            plot_and_save_losses(train_loss_history, val_loss_history, epoch, loss_plot_path)



# ==============================================================================
#                           --- 3. 脚本主入口 ---
# ==============================================================================

if __name__ == '__main__':
    print("===================================================")
    print("      STARTING MULTI-SESSION AE TRAINING           ")
    print("===================================================")
    print(f"Leaving out (1 session): {left_out_session}")
    print("---------------------------------------------------")
    
    # 调用主训练函数
    train_main_ae(config, SOURCE_SESSION, left_out_session)